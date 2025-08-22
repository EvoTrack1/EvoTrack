import math
import scipy.sparse as sp
import numpy as np
from numpy.linalg import inv
from dataset import EroTrackDataset
import json
import time

def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def compute_background_node(edges, node_types, c=0.15):

    N = len(node_types)
    k = max(1, int(N * 0.1))
    k= 5
    
    split_number = 1
    while N//split_number > 4000:
        split_number = split_number + 1
    split_edge_number = math.ceil(len(edges)/split_number)


    for j in range(split_number):
        
        if j == (split_number-1):
            edges_list = edges[j*split_edge_number :]
        else:
            edges_list = edges[j*split_edge_number:(j + 1)*split_edge_number]

        node_set = set(x for sublist in edges_list for x in sublist)
        node_list = list(node_set)

        id_idx = {m: g for g, m in enumerate(node_list)}
        idx_id = {g: m for g, m in enumerate(node_list)}

        edges_idx = []
        for edge in edges_list:
            edges_idx.append([id_idx[edge[0]], id_idx[edge[1]]])
        edges_idx = np.array(edges_idx)

        adj = sp.coo_matrix((np.ones(len(edges_list)), (edges_idx[:, 0], edges_idx[:, 1])),
                            shape=(len(node_list), len(node_list)),
                            dtype=np.float32)
        adj = adj + adj.T
        S = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_normalize(adj)).toarray())

        top_k_neighbor_intimacy_dict = {}
        for node_idx in range(len(node_list)):
            s = S[node_idx]
            s[node_idx] = -1000.0
            top_k_neighbor_idx = s.argsort()[-k:][::-1]
            top_k_neighbor_intimacy_dict[idx_id[node_idx]] = []
            for neighbor_idx in top_k_neighbor_idx:
                top_k_neighbor_intimacy_dict[idx_id[node_idx]].append([idx_id[neighbor_idx], s[neighbor_idx]])
        if j == 0:
            node_set_all = node_set
            topk_intimacy_dict = top_k_neighbor_intimacy_dict
        else:
            for node_id in top_k_neighbor_intimacy_dict.keys():
                if node_id in node_set_all:
                    unique_tuples = {}
                    for tpl in top_k_neighbor_intimacy_dict[node_id]:
                        unique_tuples[tpl[0]] = tpl
                    for tpl in topk_intimacy_dict[node_id]:
                        if tpl[0] not in unique_tuples or tpl[1] > unique_tuples[tpl[0]][1]:
                            unique_tuples[tpl[0]] = tpl
                    sorted_tuples = sorted(unique_tuples.values(), key=lambda x: x[1], reverse=True)
                    top_k = sorted_tuples[:k]
                    topk_intimacy_dict[node_id] = top_k
                else:
                    node_set_all.add(node_id)
                    topk_intimacy_dict[node_id] = top_k_neighbor_intimacy_dict[node_id]

    return topk_intimacy_dict

if __name__ == "__main__":
    
    dataname = '  '
    dataset = EroTrackDataset(dataname)
    window_number = dataset.get_time_windows_number()
    background_node = {}
    for tt in range(window_number):
        print(tt)
        node = [row[0] for row in dataset[tt]['node']]
        idx_id = {idx: id for idx, id in enumerate(node)}
        id_idx = {id: idx for idx, id in enumerate(node)}
        type = [row[1] for row in dataset[tt]['node']]
        
        subject = [id_idx[row[0]] for row in dataset[tt]['link']]
        object = [id_idx[row[1]] for row in dataset[tt]['link']]
        edge = [list(pair) for pair in zip(subject, object)]

        start_time = time.time()
        background_node[tt] = compute_background_node(edge,type)
        time_ = time.time() - start_time
        print(time_)

    with open('BackgroundNode_'+dataname+'.json', 'w') as f:
        json.dump(background_node, f, indent=2)

   