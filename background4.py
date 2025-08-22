import torch

def compute_background_embeddings3(topk_intimacy_dict, node_types, embeddings):
    
    N = len(node_types)
    device = embeddings.device
    background_embeddings = torch.zeros_like(embeddings)  
    for node_id in range(N):

        neighbors = topk_intimacy_dict.get(str(node_id), [])
        
        neighbor_ids = [n[0] for n in neighbors]
        weights = torch.tensor([n[1] for n in neighbors], device=device)
        if torch.all(weights == 0).item():
            background_embeddings[node_id] = embeddings[node_id]  
            continue
        weights = weights / weights.sum()  
        
        neighbor_embeddings = embeddings[[n for n in neighbor_ids]]
        weighted_avg = torch.sum(neighbor_embeddings * weights.view(-1, 1), dim=0)
        background_embeddings[node_id] = weighted_avg
    return background_embeddings