from collections import defaultdict
import networkx as nx
import uuid
import json
import time
import random

def remove_percent(d,k):
    keys = list(d.keys())
    num_to_remove = round(len(keys) * k)
    if num_to_remove > 0:
        to_remove = random.sample(keys, num_to_remove)
        for key in to_remove:
            del d[key]
    return d


def replace_user_path(path):
    user_parents = [
        '/home',
        '/export/home',
        '/Users',
        'C:/Users',
        'C:/Documents and Settings'
    ]
    
    original_sep = '\\' if '\\' in path else '/'
    
    normalized_path = path.replace(original_sep, '/')
    
    for parent in user_parents:
        normalized_parent = parent.replace('\\', '/')
        prefix = f"{normalized_parent}/"
        
        if normalized_path.startswith(prefix):
            suffix = normalized_path[len(prefix):]
            parts = suffix.split('/')
            
            if not parts or not parts[0]:
                continue
            
            new_suffix = '*'
            if len(parts) > 1:
                new_suffix += '/' + '/'.join(parts[1:])
            
            new_path = f"{normalized_parent}/{new_suffix}".replace('/', original_sep)
            return new_path
        
        elif normalized_path == normalized_parent:
            return path
    
    return path

def merge_similar_nodes(edges, id_type, similarity_threshold=0.5):
    graph = build_graph(edges, id_type)
    
    nodes_by_type = defaultdict(list)
    for node, node_type in id_type.items():
        nodes_by_type[node_type].append(node)
    for i, j in nodes_by_type.items():
        print(i,len(j))
    
    start_time = time.perf_counter()
    merged_nodes = {} 
    for node_type, nodes in nodes_by_type.items():
        clusters = []  
        for node in nodes:
            if not clusters:  
                clusters.append([node])
                continue
            added = False
            for cluster in clusters:
                if is_similar(node, cluster[0], graph, node_type, similarity_threshold):
                    cluster.append(node)
                    added = True
                    break
            if not added:
                clusters.append([node])  
        
        for cluster in clusters:
            super_node = f"{node_type}_super_{uuid.uuid4().hex[:8]}"  
            merged_nodes[super_node] = cluster  
            id_type[super_node] = node_type
            for original_node in cluster:
                new_edges = []
                for edge in edges:
                    if edge[0] == original_node and edge[1] == original_node:
                        continue
                    elif edge[0] == original_node:
                        new_edges.append([super_node, edge[1]])  
                    elif edge[1] == original_node:
                        new_edges.append([edge[0], super_node])  
                    else:
                        new_edges.append(edge)  
                del id_type[original_node]
                edges = new_edges  
        graph = build_graph(edges, id_type)
    end_time = time.perf_counter()
    alltime= end_time - start_time
    print(alltime)
    return edges, id_type, merged_nodes  

def is_similar(node1, node2, graph, node_type, similarity_threshold):
    
    if node_type == 2:  
        processes1 = set(graph.predecessors(node1))
        processes2 = set(graph.predecessors(node2))
    elif node_type == 0:  
        processes1 = set(graph.successors(node1))
        processes2 = set(graph.successors(node2))
    elif node_type == 1:  
        processes1 = set(graph.predecessors(node1))
        processes2 = set(graph.predecessors(node2))
    else:
        return False
    
    intersection = len(processes1 & processes2)
    union = len(processes1 | processes2)
    similarity = intersection / union if union != 0 else 0
    return similarity >= similarity_threshold

def build_graph(edges, id_type):
    
    graph = nx.DiGraph()
    for node, node_type in id_type.items():
        graph.add_node(node, type=node_type)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    return graph

def write_to_file(filename):
    with open(filename, 'w') as f:
        json.dump(superidx_attr, f, indent=2)


dataset_name = " "



link = []
with open("E:\\" + "link_raw3\\" + dataset_name, 'r') as file:
    for line in file:
        split_line = line.strip().split('\t')
        link.append(split_line)

link1 = []
for l in link:
    if 0<=int(l[7])<=429:
        link1.append(l)

link_new = []
for l in link1:
    if int(l[2]) == 1 or int(l[5]) == 1:
        continue
    link_new.append(l)



processuuid2id = {}
filepath2id = {}
socket2id = {}
id = 0
id_attr = {}
id_type = {}
link_new1 = {}
processid2name = {}

for l in link_new:
    if int(l[4]) == 0:
        if l[6] == 'EVENT_FORK' or l[6] == 'EVENT_CLONE':
            if l[0] not in processuuid2id:
                processuuid2id[l[0]] = id
                processid2name[id] = replace_user_path(l[1])
                processuuid2id[l[2]] = id
                id = id + 1
            
            elif l[3] not in processuuid2id:
                processuuid2id[l[3]] = processuuid2id[l[0]]
                    
            if processuuid2id[l[0]] not in id_type:
                id_type[processuuid2id[l[0]]] = 0
                id_attr[processuuid2id[l[0]]] = [processid2name[processuuid2id[l[0]]]]

        else:
            if l[0] not in processuuid2id:
                processuuid2id[l[0]] = id
                processid2name[id] = replace_user_path(l[1])
                id = id + 1
            if l[3] not in processuuid2id:
                processuuid2id[l[3]] = id
                processid2name[id] = replace_user_path(l[1])
                id = id + 1

            if processuuid2id[l[0]] not in id_type:
                id_type[processuuid2id[l[0]]] = 0
                id_attr[processuuid2id[l[0]]] = [processid2name[processuuid2id[l[0]]]]
            if processuuid2id[l[3]] not in id_type:
                id_type[processuuid2id[l[3]]] = 0
                id_attr[processuuid2id[l[3]]] = [processid2name[processuuid2id[l[3]]]]


            if str(processuuid2id[l[0]]) + '_'+ str(processuuid2id[l[3]]) not in link_new1:
                link_new1[str(processuuid2id[l[0]]) + '_'+ str(processuuid2id[l[3]])] = [processuuid2id[l[0]],processuuid2id[l[3]]]

    else:
        if l[0] not in processuuid2id:
            processuuid2id[l[0]] = id
            processid2name[id] = l[1]
            id = id + 1

        if processuuid2id[l[0]] not in id_type:
           id_type[processuuid2id[l[0]]] = 0 
           id_attr[processuuid2id[l[0]]] = [processid2name[processuuid2id[l[0]]]]

        if int(l[4]) == 1:
            if l[9] not in socket2id:
                socket2id[l[9]] = id
                id = id + 1

            if socket2id[l[9]] not in id_type:
                id_type[socket2id[l[9]]] = 1
                id_attr[socket2id[l[9]]] = [l[9]]


            if str(processuuid2id[l[0]]) + '_'+ str(socket2id[l[9]]) not in link_new1:
                link_new1[str(processuuid2id[l[0]]) + '_'+ str(socket2id[l[9]])] = [processuuid2id[l[0]],socket2id[l[9]]]
        
        if int(l[4]) == 2:
            pathnouser = replace_user_path(l[8])
            if pathnouser not in filepath2id:
                filepath2id[pathnouser] = id
                id = id + 1

            if filepath2id[pathnouser] not in id_type:
                id_type[filepath2id[pathnouser]] = 2
                id_attr[filepath2id[pathnouser]] = [pathnouser]

            if str(processuuid2id[l[0]]) + '_'+ str(filepath2id[pathnouser]) not in link_new1:
                link_new1[str(processuuid2id[l[0]]) + '_'+ str(filepath2id[pathnouser])] = [processuuid2id[l[0]],filepath2id[pathnouser]]
        
link_new2 = []
for l in link_new1:
    link_new2.append([link_new1[l][0],link_new1[l][1]])

print(len(link_new2), len(id_type))
id_type1 = id_type


merged_edges, merged_node_type, merged_nodes= merge_similar_nodes(link_new2, id_type, similarity_threshold=0.95)

merged_idx_type = {}
supernode_id = {}
idx = 0
merged_edges_dict = {}
for e in merged_edges:
    if e[0] not in supernode_id:
        supernode_id[e[0]] = idx
        if e[0] in merged_node_type:
            merged_idx_type[idx] = merged_node_type[e[0]]
        else:
            merged_idx_type[idx] = id_type1[e[0]]
        idx = idx + 1
    if e[1] not in supernode_id:
        supernode_id[e[1]] = idx
        if e[1] in merged_node_type:
            merged_idx_type[idx] = merged_node_type[e[1]]
        else:
            merged_idx_type[idx] = id_type1[e[1]]
        idx = idx + 1
    merged_edges_dict[str(supernode_id[e[0]]) + "_" + str(supernode_id[e[1]])] = [supernode_id[e[0]],supernode_id[e[1]]]


superidx_attr = {}
for super_node in supernode_id:
    superidx_attr[supernode_id[super_node]] = []
    for node in merged_nodes[super_node]:
        superidx_attr[supernode_id[super_node]] = superidx_attr[supernode_id[super_node]] + id_attr[node]


abstract_graph_edge = []
abstract_graph_node = []
for i in merged_edges_dict:
    abstract_graph_edge.append(merged_edges_dict[i])
for idx in merged_idx_type:
    abstract_graph_node.append([idx, merged_idx_type[idx]])

print(len(abstract_graph_edge), len(abstract_graph_node))


