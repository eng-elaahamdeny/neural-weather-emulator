import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class WeatherGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(WeatherGNN, self).__init__()
        
        # 3 graph convolution layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        
        return x


# Quick test
if __name__ == "__main__":
    # 65160 nodes, 8 input features, 64 hidden, 2 output (z at 2 pressure levels)
    model = WeatherGNN(in_channels=8, hidden_channels=64, out_channels=2)
    print(model)
    
    # Fake input to test
    x = torch.randn(65160, 8)
    edge_index = torch.randint(0, 65160, (2, 259558))
    
    out = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Model works!")
    
# This is the Graph Neural Network model — the brain of the project.
# It takes the current state of the atmosphere as input and predicts the next state 6 hours later.
# in_channels=8 — 4 variables (z, t, u, v) x 2 pressure levels = 8 features per node
# hidden_channels=64 — internal size of the network
# out_channels=2 — we predict geopotential (z) at 2 pressure levels
# GCNConv is the graph convolution layer — it lets each node gather info from its neighbors.
# 3 layers means each node can "see" up to 3 hops away — like weather influence spreading outward.