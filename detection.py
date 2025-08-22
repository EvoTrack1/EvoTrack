from dataset import EroTrackDataset
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from GCN_Module import GCN  
from GRU_Module import GRU  
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from LoadAbstractGraph import LoadAbstractGraph
from SaveHiddenState import ActiveEntityHiddenStateList
from background4 import compute_background_embeddings3
import sys  
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random
import json
from gensim.models import FastText
import os
import time

def batch_positional_encoding(pos_list, d_model=100):
    assert d_model % 2 == 0
    
    n_pos = len(pos_list)
    pe_matrix = np.zeros((n_pos, d_model))
    
    i = np.arange(d_model // 2)
    div_term = 10000 ** (-2 * i / d_model)  
    
    for row_idx, pos in enumerate(pos_list):
        angles = pos * div_term
        pe_matrix[row_idx, 0::2] = np.sin(angles)  
        pe_matrix[row_idx, 1::2] = np.cos(angles)  
    
    return pe_matrix


def get_string_vector(s,model):
    char_vectors = [model.wv[c] for c in s]
    return sum(char_vectors) / len(char_vectors)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("training_log.txt")  

fp_filename = "fp_nodes.csv"
fn_filename = "fn_nodes.csv"

with open(fp_filename, "w") as f:
    f.write("epoch,time_window,node_id,node_type,attributes\n")

with open(fn_filename, "w") as f:
    f.write("epoch,time_window,node_id,node_type,attributes\n")

def compute_metrics(outputs, labels):
    
    _, predicted = torch.max(outputs, dim=1)

    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, predicted)
    recall = recall_score(labels, predicted)  
    precision = precision_score(labels, predicted)
    f1 = f1_score(labels, predicted)
    
    tp = ((predicted == 1) & (labels == 1)).sum()  
    tn = ((predicted == 0) & (labels == 0)).sum()  
    fp = ((predicted == 1) & (labels == 0)).sum()  
    fn = ((predicted == 0) & (labels == 1)).sum()  
    if len(np.unique(labels)) >= 2:
        auc = roc_auc_score(labels, predicted)
    else:
        auc = None
    
    return accuracy, recall, precision, f1, auc, tp, tn, fp, fn

seed = 55

random.seed(seed) 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU


dataname = '  '
dataset = EroTrackDataset(dataname)

load_gcn_path = ''
load_gru_path = ''


EvolutionList_max_length = 100
ActiveEntityList_max_length = 10000000

alignment_results = {}

with open('alignment_' + dataname + '.txt', 'r') as file:  
    for line in file:
        key, value = line.strip().split()
        if value == "None":
            alignment_results[int(key)] = None
        else:
            alignment_results[int(key)] = int(value)

with open('BackgroundNode_'+dataname+'.json', 'r') as f:
    BackgroundNode = json.load(f) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
gcn = GCN(200,420).to(device)
gcn_ab = GCN(3,420).to(device)
gru = GRU(840,840,1,2).to(device) 
model_dir = "saved_models"
loaded_model = FastText.load(os.path.join(model_dir, "fasttext_model.model"))
if load_gcn_path:
    gcn.load_state_dict(torch.load(load_gcn_path, map_location=device, weights_only=True))
    print(f"Model loaded from {load_gcn_path}")
        
if load_gru_path:
    gru.load_state_dict(torch.load(load_gru_path, map_location=device, weights_only=True))
    print(f"Model loaded from {load_gru_path}")


AG_node, AG_type, AG_subject, AG_object = LoadAbstractGraph(dataname)
AG_edge = [list(pair) for pair in zip(AG_subject, AG_object)]
AG_node_types = torch.tensor([AG_type], dtype=torch.long).to(device)  
AG_node_features = F.one_hot(AG_node_types, num_classes=3).float().to(device)
AG_edge_index = torch.tensor([AG_subject, AG_object], dtype=torch.long).to(device)

AG_data = Data(x=AG_node_features, edge_index = AG_edge_index)

gcn.eval()
gru.eval()
ActiveEntityHiddenStateList_ = ActiveEntityHiddenStateList(ActiveEntityList_max_length)
for t in range(dataset.get_time_windows_number()):
    node = [row[0] for row in dataset[t]['node']]
    idx_id = {idx: id for idx, id in enumerate(node)}
    id_idx = {id: idx for idx, id in enumerate(node)}
    type = [row[1] for row in dataset[t]['node']]
    attr = [row[3] for row in dataset[t]['node']]

    subject_byid = [row[0] for row in dataset[t]['link']]
    object_byid = [row[1] for row in dataset[t]['link']]
    subject = [id_idx[row[0]] for row in dataset[t]['link']]
    object = [id_idx[row[1]] for row in dataset[t]['link']]
    edge = [list(pair) for pair in zip(subject, object)]        

    edge_weight = [row[2] for row in dataset[t]['link']]
    node_label = [row[2] for row in dataset[t]['node']]       

    node_types = torch.tensor([type], dtype=torch.long).to(device)
    edge_index = torch.tensor([subject, object], dtype=torch.long).to(device)
    edge_weights = torch.tensor(edge_weight, dtype=torch.float).to(device)
    node_label = torch.tensor(node_label).to(device)

    encoded_attr = []
    for a in attr:
        vec = get_string_vector(a,loaded_model)
        encoded_attr.append(vec)
    encoded_attr = np.array(encoded_attr)  
    encoded_attr = torch.tensor(encoded_attr, dtype=torch.float32).to(device)
    encoded_attr = encoded_attr.unsqueeze(0)
    temporal = torch.tensor(batch_positional_encoding(node), dtype=torch.float32).to(device)
    temporal = temporal.unsqueeze(0)
    node_feature = F.one_hot(node_types, num_classes=3).float().to(device)
    node_features = torch.cat((temporal, encoded_attr), dim=2)
    
    data = Data(x=node_features, edge_index = edge_index)

    start_time = time.time()
    out = gcn(data, edge_weights)
    AG_out = gcn_ab(AG_data)

    background_tt = compute_background_embeddings3(BackgroundNode[str(t)],type,out[0])

    AG_node, AG_type, AG_subject, AG_object = LoadAbstractGraph(dataname)
    AG_node_types = torch.tensor([AG_type], dtype=torch.long).to(device)  
    AG_node_features = F.one_hot(AG_node_types, num_classes=3).float().to(device)
    AG_edge_index = torch.tensor([AG_subject, AG_object], dtype=torch.long).to(device)


    active_node = ActiveEntityHiddenStateList_.get_acticeNodelsit()
    active_node_set = set(active_node)

    input = []
    inputIdx_id = {}
    idx = 0
    for i in range(len(out[0])):
        if idx_id[i] not in active_node_set:
            if alignment_results[idx_id[i]] == None:
                em = torch.cat((torch.zeros(420).to(device), torch.zeros(420).to(device)), dim=0)
                input.append(torch.stack([em], dim = 0))
            else:
                em = torch.cat((AG_out[0][alignment_results[idx_id[i]]], torch.zeros(420).to(device)), dim=0)
                input.append(torch.stack([em], dim = 0))
            inputIdx_id[idx] = idx_id[i]
            idx = idx + 1
    if input != []:        
        input = torch.stack(input, dim = 0)
        _, hidden = gru(input)  

        for inputIdx in range(input.shape[0]):
            ActiveEntityHiddenStateList_.add_node_state(inputIdx_id[inputIdx], -1, hidden[:,inputIdx,:])

    hidden_state = []
    for i in range(len(out[0])):
        hidden_state.append(ActiveEntityHiddenStateList_.get_hiddenstate(idx_id[i]))
    hidden_state = torch.stack(hidden_state, dim = 0)  
    #print(hidden_state.shape)  

    
    batch_size = 100000
    out = out.permute(1, 0, 2)

    background_ = background_tt.unsqueeze(1)
    out_expand = torch.cat((out, background_), dim=2)

    num_batches = (out_expand.size(0) + batch_size - 1) // batch_size  
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, out_expand.size(0))
        batch_part = out_expand[start_idx:end_idx]
        label_part = node_label[start_idx:end_idx]
        hidden_state_part = hidden_state[start_idx:end_idx]
        hidden_state_part = hidden_state_part.permute(1, 0, 2)
        hidden_state_part = hidden_state_part.contiguous()
        outputs, hidden = gru(batch_part, hidden_state_part)
        accuracy, recall, precision, f1, auc, tp, tn, fp, fn = compute_metrics(outputs, label_part)
        print(f"Time Window [{t}], Batch [{i}], TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
        for ii in range(hidden.shape[1]):
            idx = i*batch_size + ii
            id = idx_id[idx]
            ActiveEntityHiddenStateList_.add_node_state(id, t, hidden[:,ii,:])
   





    