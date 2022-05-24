"""
Data loader for Semeval/TACRED json files.
"""

import json
import re
import random
import torch
import numpy as np
import pickle

from utils import constant, helper, vocab

from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from flair.embeddings import TransformerWordEmbeddings

# init embedding
# embedding = TransformerWordEmbeddings("roberta-base", layers="all", layer_mean=True)
# bert_embeddings = TransformerWordEmbeddings('albert-base-v2', layers='all', layer_mean=True)
bert_embeddings = TransformerWordEmbeddings(
    "bert-large-uncased", layers="all", layer_mean=True
)


dataset = "dataset/semeval"


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v, k) for k, v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """Preprocess the data and convert to ids."""
        processed = []
        for d in data:
            tokens = list(d["token"])
            if opt["lower"]:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d["subj_start"], d["subj_end"]
            os, oe = d["obj_start"], d["obj_end"]
            tokens_ = tokens
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d["stanford_pos"], constant.POS_TO_ID)
            deprel = map_to_ids(d["stanford_deprel"], constant.DEPREL_TO_ID)
            head = [int(x) for x in d["stanford_head"]]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d["subj_start"], d["subj_end"], l)
            obj_positions = get_positions(d["obj_start"], d["obj_end"], l)
            relation = self.label2id[d["relation"]]
            processed += [
                (
                    tokens,
                    tokens_,
                    pos,
                    deprel,
                    head,
                    subj_positions,
                    obj_positions,
                    relation,
                )
            ]

        return processed

    def gold(self):
        """Return gold labels as a list."""
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """Get a batch with index."""
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        if dataset == "dataset/tacred":
            assert len(batch) == 10
        else:
            assert len(batch) == 8

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt["word_dropout"]) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors

        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)

        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        tokens_ = batch[1]
        bert_embs = get_bert_repr(tokens_, maxlen, self.opt["bert_emb_dim"])
        # print(bert_embs.shape)

        pos = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)

        # get unigrams and bigrams counts
        p_i_dict = helper.pmi_data(self.opt, n_grams="unigrams")
        p_i_j_dict = helper.pmi_data(self.opt, n_grams="bigrams")

        rand_dict_150 = pickle.load(
            open(self.opt["pmi_data_dir"] + "/rand_dict_150.p", "rb")
        )
        rand_dict_75 = pickle.load(
            open(self.opt["pmi_data_dir"] + "/rand_dict_75.p", "rb")
        )
        pmi_semeval = pickle.load(
            open(self.opt["pmi_data_dir"] + "/pmi_semeval.pkl", "rb")
        )

        rels = torch.LongTensor(batch[7])
        return (
            words,
            masks,
            pos,
            deprel,
            head,
            subj_positions,
            obj_positions,
            bert_embs,
            p_i_dict,
            p_i_j_dict,
            rand_dict_75,
            rand_dict_150,
            pmi_semeval,
            rels,
            orig_idx,
        )

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """Get subj/obj position sequence."""
    return (
        list(range(-start_idx, 0))
        + [0] * (end_idx - start_idx + 1)
        + list(range(1, length - end_idx))
    )


def get_long_tensor(tokens_list, batch_size):
    """Convert list of list of tokens to a padded LongTensor."""
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """Sort all fields by descending order of lens, and return the original indices."""
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """Randomly dropout tokens (IDs) and replace them with <UNK> tokens."""
    return [
        constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x
        for x in tokens
    ]


def get_bert_repr(tokens, maxlen, emb_dims):

    bert_embs = []

    for i in range(0, len(tokens)):

        # preprocess text - remove noisy text and characters
        text = " ".join(tokens[i])
        text = re.sub("<UNK>", "UNK", text)
        text = re.sub("-LRB-", "LRB", text)
        text = re.sub("-RRB-", "RRB", text)
        text = re.sub("'s", "s", text)
        text = re.sub("'m", "m", text)

        text = re.sub("km\\/h", "kmh", text)
        text = re.sub("Inc\\.", "Inc", text)
        text = re.sub("Rev\\.", "Rev", text)
        text = re.sub("P\\.", "P", text)
        text = re.sub("etc\\.", "etc", text)
        text = re.sub("and\\/or", "andor", text)

        # create sentence.
        sentence = Sentence(text)

        # embed a sentence using bert
        bert_embeddings.embed(sentence)

        embedding_vectors = {}
        flair_token_dict = []

        token_embs = []

        # now check out the embedded tokens.
        flair_tokens = []
        for token in sentence:
            flair_tokens.append(token)
            token_embs.append(np.asarray(token.embedding.tolist()))
        # print('token_embs: ', len(token_embs))
        # print('flair_tokens: ', flair_tokens)

        pad_terms = maxlen - len(token_embs)

        if pad_terms == -1:
            del token_embs[-1]

        if pad_terms > 0:
            for j in range(pad_terms):
                token_embs.append(np.zeros(emb_dims))

        bert_embs.append(token_embs)

    bert_embs = torch.tensor(bert_embs, dtype=torch.float)

    return bert_embs
