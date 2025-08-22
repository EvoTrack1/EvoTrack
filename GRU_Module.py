import torch.nn as nn
import torch.nn.functional as F
class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        output, h_0 = self.gru(x, hidden)
        output = self.fc(output[:,-1,:])  
        output = F.relu(output)
        return output, h_0
