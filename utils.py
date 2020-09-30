import csv
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy import sparse
import time
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a - a.T) < tol)


def read_file(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        my_dict = dict(reader)
        my_dict = {int(u): eval(v) for u, v in my_dict.items()}
        return my_dict


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    print('masking validation and test set')
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    #print(num_val, num_test)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        t = time.time()
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
        # print("test", len(test_edges_false), len(test_edges), 'time: {:.4f}s'.format(time.time() - t))

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        t = time.time()
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # if ismember([idx_i, idx_j], train_edges):
        #     continue
        # if ismember([idx_j, idx_i], train_edges):
        #     continue
        # if ismember([idx_i, idx_j], val_edges):
        #     continue
        # if ismember([idx_j, idx_i], val_edges):
        #     continue
        # if ismember([idx_i, idx_j], edges_all):
        #     continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
        # print("validation", len(val_edges_false), len(val_edges), 'time: {:.4f}s'.format(time.time() - t))

    t = time.time()
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)
    # print('time: {:.4f}s'.format(time.time() - t))

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    print('done')
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cal_sparsity(x):
    return (np.prod(x.shape) - np.count_nonzero(x)) / np.prod(x.shape)


# Load gene network dataset
'''
    Method read pandas data
    :param features : use a feature matrix 
    :param _adj : which adjacency matrix to use : homology or ontology
    :param _feat : which feature matrix to use : homology or ontology 
'''


def load_gene_data(use_features=True, _adj='homology', _feat='homology'):
    print('Loading {} dataset...'.format("Geneset"))

    if _adj == 'homology':
        adj_df = pd.read_pickle("data/ms-project/adj_homology_unweighted.txt")
    elif _adj == 'ontology':
        adj_df = pd.read_pickle("data/ms-project/adj_ontology_unweighted.txt")

    if use_features:
        if _feat == 'homology':
            feat_df = pd.read_pickle("data/ms-project/feat_homology.txt")
            feat_df_to_numpy = feat_df.to_numpy()
        elif _feat == 'ontology':
            feat_df = pd.read_pickle("data/ms-project/feat_ontology.txt")
            feat_df_to_numpy = feat_df.to_numpy()
        # print(cal_sparsity(feat_df_to_numpy))
        features_1 = scipy.sparse.csr_matrix(feat_df_to_numpy)
    else:
        size = len(list(adj_df.index.values))
        features_1 = sp.identity(len(size)).tocsr()
    features = sparse_mx_to_torch_sparse_tensor(features_1)

    adj_df = adj_df.to_numpy()
    # print(cal_sparsity(adj_df))
    # print(check_symmetric(adj_df))
    adj_df = scipy.sparse.csr_matrix(adj_df)
    adj_df = adj_df + adj_df.T.multiply(adj_df.T > adj_df) - adj_df.multiply(adj_df.T > adj_df)
    adj_df = adj_df + sp.eye(adj_df.shape[0])

    print("Finished data preprocessing")
    return adj_df, features


# get the index of the geneset in the embedding
def get_index(geneset_id):
    dict = read_file("data/ms-project/geneset_pairing.csv")
    res = dict[geneset_id]
    return res


# get the geneset at index position
def get_geneset(indicies):
    res = read_file("data/ms-project/geneset_pairing.csv")
    _dict = dict((v, k) for k, v in res.items())
    res = [_dict[i] for i in indicies]
    return res


def get_all_geneset():
    res = read_file("data/ms-project/geneset_pairing.csv")
    res = res.keys()
    return res


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return sparse_mx_to_torch_sparse_tensor(mx)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def write_file(filename, dic):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=':')
        for key, value in dic.items():
            writer.writerow([key, value])


# Point Ranking with pairwise distance
def point_ranking(point, data, indicies, genesets, top_k, metric='euclidean'):
    dist_matrix = pairwise_distances(point, data, metric=metric).tolist()
    dist_matrix = sorted(list(zip(indicies, genesets, dist_matrix[0])), key=lambda x: x[2])[0:top_k]
    return dist_matrix


# Point Ranking with KNN
def point_clustering_nn(point, cluster):
    neighbour = NearestNeighbors(n_neighbors=2)
    nbrs = neighbour.fit(cluster)
    dist, indicies = nbrs.kneighbors(cluster, return_distance=True)
    dist = np.sort(dist, axis=0)


# Setup adjacency matrix for graph
def processing():
    w_adj_dic = read_file("data/gene/adj_2.csv")
    G = nx.from_dict_of_dicts(w_adj_dic)
    x = [(u, v, d) for (u, v, d) in G.edges(data=True) if (d['weight'] > 0.0)]

    w_adj_dic = G.adj
    node_list = w_adj_dic.keys()

    adj_df = pd.DataFrame(0, index=np.arange(len(node_list)), columns=node_list)
    adj_df.insert(0, 'piv', node_list)
    adj_df.set_index('piv', inplace=True)

    w_adj_df = pd.DataFrame(0, index=np.arange(len(node_list)), columns=node_list)
    w_adj_df.insert(0, 'piv', node_list)
    w_adj_df.set_index('piv', inplace=True)

    index = 0
    for i in x:
        print(index, i[0], i[1])
        assert (G.get_edge_data(i[0], i[1]) == G.get_edge_data(i[1], i[0]))
        w_adj_df.loc[i[0], i[1]] = float(i[2]['weight'])
        w_adj_df.loc[i[1], i[0]] = float(i[2]['weight'])
        adj_df.loc[i[0], i[1]] = 1
        adj_df.loc[i[1], i[0]] = 1
        index = index + 1
    # x = w_adj_df.values
    # print(np.count_nonzero(x == 0))
    # print(np.count_nonzero(x == 1))

    adj_df.to_pickle("data/adj_data/g.txt")
    w_adj_df.to_pickle("data/adj_data/gw.txt")
    return w_adj_df


def pairwise():
    adj_df = pd.read_pickle("data/ms-project/adj_homology_unweighted.txt")
    geneset_ids = list(adj_df.index.values)
    pos = [i for i in range(len(geneset_ids))]
    dictionary = dict(zip(geneset_ids, pos))

    # Save geneset IDs for after prediction
    write_file("data/ms-project/geneset_pairing.csv", dictionary)


# reference from tensorflow cross entropy with logits
def my_cross_entropy_with_logits(labels, logits):
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = torch.where(cond, logits, zeros)
    neg_abs_logits = torch.where(cond, -logits, logits)
    x = torch.add(relu_logits - logits * labels, torch.log1p(torch.exp(neg_abs_logits)))
    return x


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

# pairwise()
