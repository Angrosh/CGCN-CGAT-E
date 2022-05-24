"""
Basic operations on trees.
"""

import numpy as np
import re
import torch


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, "_size"):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, "_depth"):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, len_):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None

    return root


def head_to_tree_dep(words, head, depvalues, len_, p_i_dict, p_i_j_dict):
    """
    Convert a sequence of head indexes into a tree object.
    """

    # convert tensor to list
    words = list(words.cpu().numpy())

    head = head[:len_].tolist()
    depvalues = depvalues[:len_].tolist()

    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):

        if (
            p_i_j_dict.get((words[i], head[i])) is None
            or p_i_dict.get(words[i]) is None
        ):
            pmi_value = 0
        else:
            pmi_value = float(p_i_j_dict.get((words[i], head[i]))) / float(
                p_i_dict.get(words[i])
            )

        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        nodes[i].dep = pmi_value

        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    return root


def tree_to_adj_dep(sent_len, tree, directed=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            if c.dep > 0:
                ret[t.idx, c.idx] = 0.255
            else:
                ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    ret = torch.from_numpy(ret)

    num_neighbours = ret.sum(dim=-1, keepdims=True)
    degs = torch.clamp(num_neighbours, min=1)
    norm = torch.pow(degs, 0.5)
    ret = torch.div(ret, norm)
    ret = ret.numpy()

    return ret


def cooccurrence_adj(sent_len, words, l, p_i_dict, p_i_j_dict):

    # convert tensor to list
    words = list(words.cpu().numpy())
    words_str = []


def tree_to_adj(sent_len, tree, directed=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1

        queue += t.children

    if not directed:
        ret = ret + ret.T

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret


def head_to_tree_prune(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h - 1]
                subj_ancestors.add(h - 1)
                h = head[h - 1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = head[h - 1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: 0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h - 1] is not None
                nodes[h - 1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adj_prune(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def head_to_tree_sdp(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h - 1]
                subj_ancestors.add(h - 1)
                h = head[h - 1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = head[h - 1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: 0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h - 1] is not None
                nodes[h - 1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root
