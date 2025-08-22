from collections import deque
import json
import re


def find_entry_nodes(edges, alarm_node):
    parent_map = {}
    for edge in edges:
        u, v = edge  
        if v not in parent_map:
            parent_map[v] = set()
        parent_map[v].add(u)
    
    visited = set()
    queue = deque([(alarm_node, 0)])
    visited.add(alarm_node)
    entry_nodes = set()
    entry_nodes_depth = {}
    
    while queue:
        node, depth= queue.popleft()
        if node not in parent_map or not parent_map[node]:
            entry_nodes.add(node)
            if depth not in entry_nodes_depth:
                entry_nodes_depth[depth] = 1
            else:
                entry_nodes_depth[depth] = entry_nodes_depth[depth] + 1
        else:
            for parent in parent_map[node]:
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth+1))
    
    return entry_nodes


def find_entry_nodes_update(edges, alignment, neigh, alarm_node, delta):

    parent_map = {}
    for start, end in edges:
        if end not in parent_map:
            parent_map[end] = set()
        parent_map[end].add(start)
    
    visited = set([alarm_node])  
    queue = deque([(alarm_node, 0, [0])])  
    entry_nodes = set()  
    depth_list_set = {}
    depth_list_set1 = {}

    while queue:
        node, depth, depth_list = queue.popleft()
        
        if node not in parent_map or not parent_map[node]:
            entry_nodes.add(node)
            if ''.join(map(str, depth_list)) not in depth_list_set:
                depth_list_set[''.join(map(str, depth_list))] = 1 
            else:
                depth_list_set[''.join(map(str, depth_list))] = depth_list_set[''.join(map(str, depth_list))] + 1
            continue

        elif depth >= delta:
            if ''.join(map(str, depth_list)) not in depth_list_set1:
                depth_list_set1[''.join(map(str, depth_list))] = 1
            else:
                depth_list_set1[''.join(map(str, depth_list))] = depth_list_set1[''.join(map(str, depth_list))] + 1
            continue
        
        else:
            if node in alignment:
                align_node = alignment[node]
                for parent in parent_map[node]:
                    if parent in alignment and alignment[parent] in neigh[align_node]:
                        if parent not in visited:
                            visited.add(parent)
                            depth_list1  = depth_list.copy()
                            depth_list1.append(depth + 1)
                            queue.append((parent, depth + 1, depth_list1))
                    else:
                        if parent not in visited:
                            visited.add(parent)
                            depth_list1  = depth_list.copy()
                            depth_list1.append(0)
                            queue.append((parent, 0, depth_list1))
            else:
                for parent in parent_map[node]:
                    if parent not in visited:
                        visited.add(parent)
                        depth_list1  = depth_list.copy()
                        depth_list1.append(0)
                        queue.append((parent, 0, depth_list1))
    return entry_nodes


def read_from_file(filename):
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
        converted_data = {int(k): v for k, v in loaded_data.items()}
    return converted_data

dataset_name = " "

link1_loaded = []
with open('edge_list.csv', 'r') as f:
    next(f)  
    for line in f:
        source, target = line.strip().split(',')
        link1_loaded.append([int(source), int(target)])


with open("AbstractGraph/" + dataset_name + "/edge", 'r') as ef:
    edge_list = []
    for line in ef:
        parts = re.split(r'\s+', line.strip())
        parent, child = int(parts[0]), int(parts[1])
        edge_list.append((parent, child))

neigh = {}
for edge in edge_list:
    u, v = edge  
    if v not in neigh:
        neigh[v] = []
    if u not in neigh:
        neigh[u] = []
    neigh[v].append(u)
    neigh[u].append(v)

for node in neigh:
    neigh[node] = set(neigh[node])

alignment = {}
with open('alignment_r_' + dataset_name + '.txt', 'r') as file:  
    for line in file:
        key, value = line.strip().split()
        alignment[int(key)] = int(value)


alarm_node = [1812, 2349096, 2382145, 2384653, 2386134, 929010, 932404, 925537, 925986, 925538, 925987, 927336]

with open('entry_nodes_results.txt', 'w') as f, open('entry_nodes_length.csv', 'w') as f_length:
    for a in alarm_node:
        entry_nodes_update1 = find_entry_nodes_update(link1_loaded, alignment, neigh, a, 1)
        entry_nodes_update2 = find_entry_nodes_update(link1_loaded, alignment, neigh, a, 2)
        entry_nodes_update3 = find_entry_nodes_update(link1_loaded, alignment, neigh, a, 3)
        entry_nodes_update4 = find_entry_nodes_update(link1_loaded, alignment, neigh, a, 4)
        entry_nodes_update5 = find_entry_nodes_update(link1_loaded, alignment, neigh, a, 5)

        str_up2 = ','.join(map(str, sorted(entry_nodes_update2)))
        str_up3 = ','.join(map(str, sorted(entry_nodes_update3)))
        str_up4 = ','.join(map(str, sorted(entry_nodes_update4)))
        str_up5 = ','.join(map(str, sorted(entry_nodes_update5)))
        
        len_up2 = len(entry_nodes_update2)
        len_up3 = len(entry_nodes_update3)
        len_up4 = len(entry_nodes_update4)
        len_up5 = len(entry_nodes_update5)


