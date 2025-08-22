import networkx as nx
from collections import defaultdict
import re
from dataset import EroTrackDataset
import json


def read_from_file(filename):
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
        converted_data = {int(k): v for k, v in loaded_data.items()}
    return converted_data


def initialize_similarity(G1, G2, sim, alignment):
    list = []
    v1_set = set()
    for v1 in G1.nodes():
        if v1 not in alignment:
            v1_set.add(v1)
            for v2 in G2.nodes():
                if G1.nodes[v1]['type'] == G2.nodes[v2]['type']:
                    list.append((v1,v2))
                    sim[v1][v2] = 1.0
                else:
                    sim[v1][v2] = 0.0
    return sim, list, v1_set


def semantics_alignment(G1, G2, attr, supernode_attr, sim, alignment):
    for v1 in G1.nodes():
        if v1 not in alignment:
            for v2 in G2.nodes():
                if G1.nodes[v1]['type'] != G2.nodes[v2]['type']:
                    continue
                assigned = 0
                for s in supernode_attr[v2]:
                    if attr[v1] == s:
                        assigned = 1
                        alignment[v1] = v2
                        for vv2 in G2.nodes():
                            if G1.nodes[v1]['type'] == G2.nodes[vv2]['type']:
                                sim[v1][vv2] = 1
                            else:
                                sim[v1][vv2] = 0
                        sim[v1][v2] = 2
                        break
                if assigned == 1:
                    break
    return  alignment, sim

def structural_alignment(G1, G2, sim, alignment, list1):

    sim, list, v1_set= initialize_similarity(G1, G2, sim, alignment)
    
        
    for l in list:
        v1 = l[0]
        v2 = l[1]
        total_score = 0.0
        type_count = 0
        
        for node_type in [2, 0, 1]:
            neighbors_v1 = [u for u in G1.neighbors(v1) if G1.nodes[u]['type'] == node_type]
            neighbors_v2 = [u for u in G2.neighbors(v2) if G2.nodes[u]['type'] == node_type]
            
            score = 0.0
            matched_pairs = []
            used_v2 = set()
            
            for u1 in sorted(neighbors_v1, 
                        key=lambda x: max([sim[x].get(u2, 0) for u2 in neighbors_v2], default=0),  
                        reverse=True): 
                best_match = None
                best_sim = 0
                
                for u2 in neighbors_v2:
                    if u2 not in used_v2 and sim[u1].get(u2, 0) > best_sim:
                        best_sim = sim[u1][u2]
                        best_match = u2
                        
                if best_match is not None:
                    matched_pairs.append((u1, best_match))
                    used_v2.add(best_match)
                    score += best_sim
            
            max_neighbors = len(neighbors_v1)
            if max_neighbors > 0:
                score /= max_neighbors
            
            total_score += score
            if neighbors_v1 !=[] and neighbors_v2 !=[]:
                type_count += 1

        avg_score = total_score / type_count if type_count > 0 else 0
        sim[v1][v2] = avg_score
                
    for v1 in v1_set:
        best_v2 = max(G2.nodes(), 
                     key=lambda v2: sim[v1].get(v2, 0) if G1.nodes[v1]['type'] == G2.nodes[v2]['type'] else -1)
        alignment[v1] = best_v2 if sim[v1].get(best_v2, 0) > 0 else None

    for v1 in v1_set:
        list = []
        for v2 in G2:
            if G1.nodes[v1]['type'] == G2.nodes[v2]['type']:
                list.append(sim[v1][v2])
        list1.append(list)

    return alignment, sim, list1 


def build_graph_byfile(node_file, edge_file):
    graph = nx.Graph()

    with open(node_file, 'r') as nf:
        node_list = []
        for line in nf:
            parts = re.split(r'\s+', line.strip())
            node, node_type = int(parts[0]), int(parts[1])
            node_list.append((node,{'type':node_type}))

    with open(edge_file, 'r') as ef:
        edge_list = []
        for line in ef:
            parts = re.split(r'\s+', line.strip())
            parent, child = int(parts[0]), int(parts[1])
            edge_list.append((parent, child))
    graph.add_nodes_from(node_list)   
    graph.add_edges_from(edge_list)

    return graph

import networkx as nx

def build_graph_bylist(nodeid, nodetype, subject, object):
    graph = nx.Graph()
    
    graph.add_nodes_from(
        (nid, {"type": ntype, "neighbor": []}) 
        for nid, ntype in zip(nodeid, nodetype)
    )
    
    for src, tgt in zip(subject, object):
        graph.add_edge(src, tgt)
        
        graph.nodes[src].setdefault('neighbor', []).append(tgt)
        graph.nodes[tgt].setdefault('neighbor', []).append(src)
    
    return graph


if __name__ == "__main__":


    dataname = ' '
    dataset = EroTrackDataset(dataname)
    window_number = dataset.get_time_windows_number()
    sim = defaultdict(dict)
    alignment = {}

    supernode_attr= read_from_file("AbstractGraph/" + dataname + "/attr")

    G2 = build_graph_byfile("AbstractGraph/" + dataname + "/node", "AbstractGraph/" + dataname + "/edge")
    list1 = []
    for tt in range(window_number):
        print(tt)
        node = [row[0] for row in dataset[tt]['node']]
        type = [row[1] for row in dataset[tt]['node']]
        a = [row[3] for row in dataset[tt]['node']]
        attr = {}
        for i in range(len(node)):
            attr[node[i]] = a[i]

        subject = [row[0] for row in dataset[tt]['link']]
        object = [row[1] for row in dataset[tt]['link']]

        G1 = build_graph_bylist(node, type, subject, object)

        alignment, sim = semantics_alignment(G1, G2, attr, supernode_attr, sim, alignment)
        alignment, sim, list1 = structural_alignment(G1, G2, sim, alignment, list1)


    with open('alignment_' + dataname + '.txt', 'w') as f:
        for key, value in alignment.items():
            f.write(f"{key} {value}\n")
