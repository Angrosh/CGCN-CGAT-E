"""
CGCN-CGATE model for relation extraction.
"""
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch.autograd import Variable

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops, add_remaining_self_loops
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_scatter import scatter, scatter_add

from collections import defaultdict
from math import log

from model.tree import (
    Tree,
    head_to_tree,
    tree_to_adj,
    head_to_tree_dep,
    tree_to_adj_dep,
    cooccurrence_adj,
)
from model.tree import head_to_tree_prune, tree_to_adj_prune

from utils import constant, torch_utils


class Edge_GATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,  # new
        heads=1,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
    ):
        super(Edge_GATConv, self).__init__(aggr="add")  # "Add" aggregation.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim  # new
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels)
        )  # emb(in) x [H*emb(out)]
        self.att = Parameter(
            torch.Tensor(1, heads, 2 * out_channels + edge_dim)
        )  # 1 x H x [2*emb(out)+edge_dim]    # new

        self.edge_update = Parameter(
            torch.Tensor(out_channels + edge_dim, out_channels)
        )  # [emb(out)+edge_dim] x emb(out)  # new

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)  # new
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        # 1. Linearly transform node feature matrix (XΘ)
        x = torch.mm(x, self.weight).view(
            -1, self.heads, self.out_channels
        )  # N x H x emb(out)

        # 2. Add self-loops to the adjacency matrix (A' = A + I)
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)  # 2 x E
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # 2 x (E+N)

        # 2.1 Add node's self information (value=0) to edge_attr
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(
            edge_index.device
        )  # N x edge_dim   # new
        edge_attr = torch.cat(
            [edge_attr, self_loop_edges], dim=0
        )  # (E+N) x edge_dim  # new

        edge_attr1 = torch.ones([7], dtype=torch.float)

        # 3. Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        return x_j * alpha.view(-1, self.heads, 1)  # (E+N) x H x (emb(out)+edge_dim)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = torch.mm(aggr_out, self.edge_update)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GCNClassifier(nn.Module):
    """A wrapper classifier for GCNRelationModel."""

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt["hidden_dim"]
        self.classifier = nn.Linear(in_dim, opt["num_class"])
        self.opt = opt

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(
            opt["vocab_size"], opt["emb_dim"], padding_idx=constant.PAD_ID
        )
        self.pos_emb = (
            nn.Embedding(len(constant.POS_TO_ID), opt["pos_dim"])
            if opt["pos_dim"] > 0
            else None
        )
        # self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = CGCNCGAT(opt, embeddings)

        # mlp output layer
        in_dim = opt["hidden_dim"] * 3
        layers = [nn.Linear(in_dim, opt["hidden_dim"]), nn.ReLU()]
        for _ in range(self.opt["mlp_layers"] - 1):
            layers += [nn.Linear(opt["hidden_dim"], opt["hidden_dim"]), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt["topn"] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt["topn"] < self.opt["vocab_size"]:
            print("Finetune top {} word embeddings.".format(self.opt["topn"]))
            self.emb.weight.register_hook(
                lambda x: torch_utils.keep_partial_grad(x, self.opt["topn"])
            )
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        (
            words,
            masks,
            pos,
            deprel,
            head,
            subj_pos,
            obj_pos,
            bert_embs,
            p_i_dict,
            p_i_j_dict,
            rand_dict_75,
            rand_dict_150,
            pmi_semeval,
        ) = inputs  # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        # convert deprel tensor to list
        deprel_list = deprel.cpu().numpy().tolist()
        new_deprels_list = []
        for deps_list in deprel_list:
            new_deprels_list.append([item for item in deps_list if item != 0])

        words = words.cpu().numpy().tolist()
        dep_rels = deprel.cpu().numpy().tolist()
        heads = head.cpu().numpy().tolist()

        source, target, source1, target1 = [], [], [], []

        node_index_dict = defaultdict(int)
        for i in range(0, len(words)):

            words_list = words[i][: l[i]]
            head_list = heads[i][: l[i]]
            deprel_list = dep_rels[i][: l[i]]

            source_tmp, target_tmp = [], []
            j = 0
            while j < len(words_list):
                if words_list[j] not in node_index_dict:
                    node_index_dict[words_list[j]] = len(node_index_dict)

                if head_list[j] < len(words_list):
                    if words_list[j] != words_list[head_list[j]]:
                        source_tmp.append(words_list[j])
                        source1.append(words_list[j])

                        target_tmp.append(words_list[head_list[j]])
                        target1.append(words_list[head_list[j]])

                j = j + 1

            for term in source_tmp:
                source.append(node_index_dict.get(term))
            for term in target_tmp:
                target.append(node_index_dict.get(term))

        edge_weights = []

        j = 0
        while j < len(source1):
            pij = p_i_j_dict.get((source1[j], target1[j]))
            pi = p_i_dict.get((source1[j]))
            pj = p_i_dict.get((target1[j]))
            if pij is not None and pi is not None and pj is not None:
                pmi = round(log(pij / pi * pj), 1)
                if pmi > 0:
                    edge_weights.append([pmi])
                else:
                    edge_weights.append([0])
            else:
                edge_weights.append([0])

            j = j + 1

        tmp = []
        tmp.append(source)
        tmp.append(target)
        edge_index = torch.from_numpy(np.array(tmp)).type(torch.LongTensor)
        edge_index_gat = Variable(edge_index.cuda())

        edge_weight = torch.from_numpy(np.array(edge_weights)).type(torch.LongTensor)
        edge_weight = Variable(edge_weight.cuda())

        def inputs_to_tree_reps_prune(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = (
                head.cpu().numpy(),
                words.cpu().numpy(),
                subj_pos.cpu().numpy(),
                obj_pos.cpu().numpy(),
            )
            trees = [
                head_to_tree_prune(
                    head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]
                )
                for i in range(len(l))
            ]
            adj = [
                tree_to_adj_prune(
                    maxlen, tree, directed=False, self_loop=False
                ).reshape(1, maxlen, maxlen)
                for tree in trees
            ]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt["cuda"] else Variable(adj)

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [
                tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen)
                for tree in trees
            ]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt["cuda"] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, l)
        h, pool_mask = self.gcn(adj, inputs, edge_index_gat, edge_weight)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(
            0
        ).unsqueeze(
            2
        )  # invert mask
        pool_type = self.opt["pooling"]
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type="max")
        obj_out = pool(h, obj_mask, type="max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)

        return outputs, h_out


class CGCNCGAT(nn.Module):
    def __init__(self, opt, embeddings):
        super().__init__()
        self.opt = opt
        # word embeddings
        # self.in_dim = opt['emb_dim'] + opt['pos_dim']
        self.emb, self.pos_emb = embeddings
        self.use_cuda = opt["cuda"]
        self.mem_dim = opt["hidden_dim"]
        self.bert_dim = 1024
        self.in_dim = self.bert_dim + opt["pos_dim"]

        # rnn layer
        if self.opt.get("rnn", False):
            self.input_W_R = nn.Linear(self.in_dim, opt["rnn_hidden"])

            self.rnn = nn.LSTM(
                opt["rnn_hidden"],
                opt["rnn_hidden"],
                opt["rnn_layers"],
                batch_first=True,
                dropout=opt["rnn_dropout"],
                bidirectional=True,
            )
            self.in_dim = opt["rnn_hidden"] * 2
            self.rnn_drop = nn.Dropout(opt["rnn_dropout"])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(opt["input_dropout"])
        self.num_layers = opt["num_layers"]

        self.layers = nn.ModuleList()

        self.sublayer_first = opt["sublayer_first"]
        self.sublayer_second = opt["sublayer_second"]

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(
                    GraphConvLayer(opt, self.mem_dim, self.sublayer_first)
                )
                self.layers.append(
                    GraphConvLayer(opt, self.mem_dim, self.sublayer_second)
                )

        self.aggregate_W = nn.Linear(
            len(self.layers) * self.mem_dim + self.mem_dim, self.mem_dim
        )

        self.gcn_drop = nn.Dropout(opt["gcn_dropout"])

        self.gat_hidden = int(self.mem_dim / 2)

        self.edgegatconv1 = Edge_GATConv(self.mem_dim, self.gat_hidden, 1)
        self.edgegatconv2 = Edge_GATConv(
            self.mem_dim + self.gat_hidden, self.gat_hidden, 1
        )

        self.edgegatconv11 = Edge_GATConv(self.mem_dim, self.gat_hidden, 1)
        self.edgegatconv22 = Edge_GATConv(
            self.mem_dim + self.gat_hidden, self.gat_hidden, 1
        )

        self.Linear = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(
            batch_size, self.opt["rnn_hidden"], self.opt["rnn_layers"]
        )
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(
            rnn_inputs, seq_lens, batch_first=True
        )
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, edge_index_gat, edge_weight):
        (
            words,
            masks,
            pos,
            deprel,
            head,
            subj_pos,
            obj_pos,
            bert_embs,
            p_i_dict,
            p_i_j_dict,
            rand_dict_75,
            rand_dict_150,
            pmi_semeval,
        ) = inputs  # unpack

        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)

        word_embs = self.emb(words)
        # embs = [word_embs]
        embs = [bert_embs]

        # print(bert_embs.shape)
        # print()

        if self.opt["pos_dim"] > 0:
            embs += [self.pos_emb(pos)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        if self.opt.get("rnn", False):
            embs = self.input_W_R(embs)
            gcn_inputs = self.rnn_drop(
                self.encode_with_rnn(embs, masks, words.size()[0])
            )
        else:
            gcn_inputs = embs
        gcn_inputs = self.input_W_G(gcn_inputs)

        layer_list = []
        outputs = gcn_inputs

        for i in range(len(self.layers)):
            if i < 2:
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)

        gat_blocks_list = []

        for j in range(2):
            if j == 0:
                outputs = gcn_inputs
                cache_list = [outputs]
                output_list = []
                for i in range(1, 3):

                    outputs = outputs.view(-1, outputs.shape[2])

                    if i == 1:
                        x = F.relu(
                            self.edgegatconv1(outputs, edge_index_gat, edge_weight)
                        )
                    if i == 2:
                        x = F.relu(
                            self.edgegatconv2(outputs, edge_index_gat, edge_weight)
                        )

                    x = x.reshape(
                        gcn_inputs.shape[0], gcn_inputs.shape[1], self.gat_hidden
                    )
                    cache_list.append(x)
                    outputs = torch.cat(cache_list, dim=2)
                    output_list.append(self.gcn_drop(x))

                gcn_outputs = torch.cat(output_list, dim=2)
                gcn_outputs = gcn_outputs + gcn_inputs
                gat_blocks_list.append(gcn_outputs)

            if j == 1:
                outputs = gcn_inputs
                cache_list = [outputs]
                output_list = []
                for i in range(1, 3):

                    outputs = outputs.view(-1, outputs.shape[2])

                    if i == 1:
                        x = F.relu(
                            self.edgegatconv11(outputs, edge_index_gat, edge_weight)
                        )
                    if i == 2:
                        x = F.relu(
                            self.edgegatconv22(outputs, edge_index_gat, edge_weight)
                        )

                    x = x.reshape(
                        gcn_inputs.shape[0], gcn_inputs.shape[1], self.gat_hidden
                    )
                    cache_list.append(x)
                    outputs = torch.cat(cache_list, dim=2)
                    output_list.append(self.gcn_drop(x))

                gcn_outputs = torch.cat(output_list, dim=2)
                gcn_outputs = gcn_outputs + gcn_inputs
                gat_blocks_list.append(gcn_outputs)

        final_output = torch.cat(gat_blocks_list, dim=2)
        out = self.Linear(final_output)
        layer_list.append(out)

        aggregate_out = torch.cat(layer_list, dim=2)

        dcgcn_output = self.aggregate_W(aggregate_out)
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        return dcgcn_output, mask


class GraphConvLayer(nn.Module):
    """A GCN module operated on dependency graphs."""

    def __init__(self, opt, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt["gcn_dropout"])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(
                nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim)
            )

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

        self.c_in = 360
        self.c_out = 320
        self.num_heads = 1
        concat_heads = True
        alpha = 0.2

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(self.c_in, self.c_out * self.num_heads)
        self.a = nn.Parameter(
            torch.Tensor(self.num_heads, 2 * self.c_out)
        )  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out


def pool(h, mask, type="max"):
    if type == "max":
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == "avg":
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def softmax(src, index, num_nodes):
    """
    Given a value tensor: `src`, this function first groups the values along the first dimension
    based on the indices specified in: `index`, and then proceeds to compute the softmax individually for each group.
    """
    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    out = src - scatter(src, index, dim=0, dim_size=N, reduce="max")[index]
    out = out.exp()
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce="sum")[index]
    return out / (out_sum + 1e-16)
