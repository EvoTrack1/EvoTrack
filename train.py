from dataset import EroTrackDataset
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from ActiveEntityList import ActiveEntityList
from GCN_Module import GCN  
from GRU_Module import GRU  
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from LoadAbstractGraph import LoadAbstractGraph
from background4 import compute_background_embeddings3
import math
import sys  
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random
import json
from gensim.models import FastText
import os


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
    

def get_last_chars(s, n):
    return s[-n:] if s is not None else ""


seed = 42

random.seed(seed) 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  

dataname = ' '
dataset = EroTrackDataset(dataname)

load_gcn_path = ' '
load_gcn2_path = ' '
load_gru_path = ' '
save_gcn_path = ' '
save_gcn2_path = ' '
save_gru_path = ' '

num_epochs = 10
lr_GCN = 0.001
lr_gru = 0.001

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
gcn = GCN(200,420).to(device)
gcn_ab = GCN(3,420).to(device)
gru = GRU(840,840,1,2).to(device) 
model_dir = "saved_models"
loaded_model = FastText.load(os.path.join(model_dir, "fasttext_model.model"))

if load_gcn_path:
    gcn.load_state_dict(torch.load(load_gcn_path, map_location=device))
    print(f"Model loaded from {load_gcn_path}")
        
if load_gcn2_path:
    gcn_ab.load_state_dict(torch.load(load_gcn2_path, map_location=device))
    print(f"Model loaded from {load_gcn2_path}")
    
if load_gru_path:
    gru.load_state_dict(torch.load(load_gru_path, map_location=device))
    print(f"Model loaded from {load_gru_path}")


optimizer = torch.optim.Adagrad([
{'params': gcn.parameters(), 'lr': lr_GCN, 'weight_decay': 1e-4},
{'params': gcn_ab.parameters(), 'lr': lr_GCN, 'weight_decay': 1e-4},
{'params': gru.parameters(), 'lr': lr_gru, 'weight_decay': 1e-4},
])


#加载抽象图
AG_node, AG_type, AG_subject, AG_object = LoadAbstractGraph(dataname)
AG_edge = [list(pair) for pair in zip(AG_subject, AG_object)]
AG_node_types = torch.tensor([AG_type], dtype=torch.long).to(device)  
AG_node_features = F.one_hot(AG_node_types, num_classes=3).float().to(device)
AG_edge_index = torch.tensor([AG_subject, AG_object], dtype=torch.long).to(device)

AG_data = Data(x=AG_node_features, edge_index = AG_edge_index)


for epoch in range(num_epochs):
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    all_outputs = []
    all_labels = []
    for ii in range(4):
        for t in range(1+(5*ii), (1+5*(ii+1))):
            ActiveEntityList_ = ActiveEntityList(EvolutionList_max_length, ActiveEntityList_max_length)
            node = []
            for tt in range(1+(5*ii),t+1):
                node = [row[0] for row in dataset[tt]['node']]
                idx_id = {idx: id for idx, id in enumerate(node)}
                id_idx = {id: idx for idx, id in enumerate(node)}
                type = [row[1] for row in dataset[tt]['node']]
                attr = [row[3] for row in dataset[tt]['node']]

                subject_byid = [row[0] for row in dataset[tt]['link']]
                object_byid = [row[1] for row in dataset[tt]['link']]
                subject = [id_idx[row[0]] for row in dataset[tt]['link']]
                object = [id_idx[row[1]] for row in dataset[tt]['link']]
                edge = [list(pair) for pair in zip(subject, object)]
                
                edge_weight = [row[2] for row in dataset[tt]['link']]
                node_label = [row[2] for row in dataset[tt]['node']]

                node_types = torch.tensor([type], dtype=torch.long).to(device)
                edge_index = torch.tensor([subject, object], dtype=torch.long).to(device)
                edge_weights = torch.tensor(edge_weight, dtype=torch.float).to(device)
              
                encoded_attr = []
                n = 20
                for a in attr:
                    vec = get_string_vector(get_last_chars(a,n),loaded_model)
                    encoded_attr.append(vec)
                encoded_attr = np.array(encoded_attr)  
                encoded_attr = torch.tensor(encoded_attr, dtype=torch.float32).to(device)
                encoded_attr = encoded_attr.unsqueeze(0)

                temporal = torch.tensor(batch_positional_encoding(node), dtype=torch.float32).to(device)
                temporal = temporal.unsqueeze(0)

                node_feature = F.one_hot(node_types, num_classes=3).float().to(device)
                node_features = torch.cat((temporal, encoded_attr), dim=2)

                data = Data(x=node_features, edge_index = edge_index)
                out = gcn(data, edge_weights)
                AG_out = gcn_ab(AG_data)

                background_tt = compute_background_embeddings3(BackgroundNode[str(tt)],type,out[0])

                update_node = set()
                active_node = ActiveEntityList_.get_acticeNodelsit()
                active_node_set = set(active_node)
                for i in range(len(out[0])):
                    update_node.add(idx_id[i])
                    if idx_id[i] not in active_node_set:
                        if alignment_results[idx_id[i]] == None:
                            em = torch.cat((torch.zeros(420).to(device), torch.zeros(420).to(device)), dim=0)
                            ActiveEntityList_.add_node_state(idx_id[i], -1, em, 0)
                        else:
                            em = torch.cat((AG_out[0][alignment_results[idx_id[i]]], torch.zeros(420).to(device)), dim=0)
                            ActiveEntityList_.add_node_state(idx_id[i], -1, em, 0)
                    em = torch.cat((out[0][i], background_tt[i]), dim=0)
                    ActiveEntityList_.add_node_state(idx_id[i], tt-(1+5*ii), em, node_label[i])

                active_node = ActiveEntityList_.get_acticeNodelsit()
                for id in active_node:
                    if id not in update_node:
                        ActiveEntityList_.add_node_state(id, tt-(1+5*ii))

            batch = []
            label = []
            for id in node:    
                a = ActiveEntityList_.get_Evolution(id).EntityEvolutionList.to_list()
                sequence = torch.stack(ActiveEntityList_.get_Evolution(id).EntityEvolutionList.to_list(), dim = 0)
                batch.append(sequence)
                label.append(ActiveEntityList_.get_Evolution(id).label)

            batch = torch.stack(batch, dim = 0).to(device)  
            label = torch.tensor(label).to(device)

            if ii >= 3:
                gcn.eval()
                gru.eval()
                batch_size = 1000000
                num_batches = (batch.size(0) + batch_size - 1) // batch_size  

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, batch.size(0))
                    batch_part = batch[start_idx:end_idx]
                    label_part = label[start_idx:end_idx]
                    

                    outputs, _ = gru(batch_part)
                    accuracy, recall, precision, f1, auc, tp, tn, fp, fn = compute_metrics(outputs, label_part)

                    total_tp += tp
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn
                    all_outputs.append(outputs.detach().cpu())  
                    all_labels.append(label_part.detach().cpu())
                    
                    predicted = outputs.argmax(dim=1).cpu().numpy()
                    labels_np = label_part.cpu().numpy()
                    fp_mask = (predicted == 1) & (labels_np == 0)
                    fn_mask = (predicted == 0) & (labels_np == 1)
                    fp_indices = np.where(fp_mask)[0]
                    fn_indices = np.where(fn_mask)[0]
                    
                    current_node_ids = node[start_idx:end_idx]
                    current_types = type[start_idx:end_idx]
                    current_attr = attr[start_idx:end_idx]
                    
                    for idx in fp_indices:
                        global_idx = start_idx + idx
                        node_id = current_node_ids[idx]
                        node_type = current_types[idx]
                        attributes = attr[idx]
                        with open(fp_filename, "a") as f:
                            f.write(f"{epoch},{t},{node_id},{node_type},{attributes}\n")

                    for idx in fn_indices:
                        global_idx = start_idx + idx
                        node_id = current_node_ids[idx]
                        node_type = current_types[idx]
                        attributes = attr[idx]
                        with open(fn_filename, "a") as f:
                            f.write(f"{epoch},{t},{node_id},{node_type},{attributes}\n")    
                    print(f"Epoch [{epoch}], Time Window [{t}], Batch [{i}], TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            
            if ii < 3:
                gcn.train()
                gru.train()
         
                batch_size = 1000000
                num_batches = (batch.size(0) + batch_size - 1) // batch_size  
                for i in range(num_batches):
                    optimizer.zero_grad()
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, batch.size(0))
                    batch_part = batch[start_idx:end_idx]
                    label_part = label[start_idx:end_idx]
                    
                    num_zeros = (label_part == 0).sum().item()
                    num_ones = (label_part == 1).sum().item()
                    weight_0 = math.pow(num_ones, 1)
                    weight_1 = math.pow(num_zeros, 1)
                    sum_weights = weight_0 + weight_1
                    normalized_weight_0 = weight_0 / sum_weights
                    normalized_weight_1 = weight_1 / sum_weights
                    class_weights = torch.tensor([normalized_weight_0, normalized_weight_1]).to(device)
                    if (class_weights == 0).any():
                        class_weights[class_weights == 0] = 1.0  
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    
                    outputs, _ = gru(batch_part)
                    loss = criterion(outputs, label_part)
                    loss.backward()
                    optimizer.step()

                    accuracy, recall, precision, f1, auc, tp, tn, fp, fn = compute_metrics(outputs, label_part)

                    predicted = outputs.argmax(dim=1).cpu().numpy()
                    labels_np = label_part.cpu().numpy()
                    fp_mask = (predicted == 1) & (labels_np == 0)
                    fn_mask = (predicted == 0) & (labels_np == 1)
                    fp_indices = np.where(fp_mask)[0]
                    fn_indices = np.where(fn_mask)[0]
                    
                    current_node_ids = node[start_idx:end_idx]
                    current_types = type[start_idx:end_idx]
                    current_attr = attr[start_idx:end_idx]
                    
                    for idx in fp_indices:
                        global_idx = start_idx + idx
                        node_id = current_node_ids[idx]
                        node_type = current_types[idx]
                        attributes = attr[idx]
                        with open(fp_filename, "a") as f:
                            f.write(f"{epoch},{t},{node_id},{node_type},{attributes}\n")

                    for idx in fn_indices:
                        global_idx = start_idx + idx
                        node_id = current_node_ids[idx]
                        node_type = current_types[idx]
                        attributes = attr[idx]
                        with open(fn_filename, "a") as f:
                            f.write(f"{epoch},{t},{node_id},{node_type},{attributes}\n")        
                    
                    print(f"Epoch [{epoch}], Time Window [{t}], Batch [{i}], "
                          f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1: {f1},  auc: {auc}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, Loss: {loss}")
                 
    total_samples = total_tp + total_tn + total_fp + total_fn
    total_accuracy = (total_tp + total_tn) / total_samples if total_samples != 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) != 0 else 0
    
    if all_outputs:
        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        probabilities = torch.softmax(all_outputs_tensor, dim=1)[:, 1].numpy()  
        labels_np = all_labels_tensor.numpy()
        from sklearn.metrics import roc_auc_score
        total_auc = roc_auc_score(labels_np, probabilities)
    else:
        total_auc = 0.0
    
    print(f"\nEpoch [{epoch}], Time Window [{t}], Total Metrics: "
        f"Accuracy: {total_accuracy:.4f}, Recall: {total_recall:.4f}, Precision: {total_precision:.4f}, "
        f"F1: {total_f1:.4f},  AUC: {total_auc:.4f}, TP: {total_tp}, TN: {total_tn}, FP: {total_fp}, FN: {total_fn}\n")
    
if save_gcn_path:
    torch.save(gcn.state_dict(), save_gcn_path)
    print(f"Model saved to {save_gcn_path}")

if save_gcn2_path:
    torch.save(gcn_ab.state_dict(), save_gcn2_path)
    print(f"Model saved to {save_gcn2_path}")

if save_gru_path:
    torch.save(gru.state_dict(), save_gru_path)    
    print(f"Model saved to {save_gru_path}")                    
   
    
