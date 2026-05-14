import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(8, 32, heads=4, concat=True)
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.gat2 = GATConv(128, 64, heads=4, concat=True)
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, data, return_embedding=False):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))

        if return_embedding:
            return embedding

        out = self.fc3(embedding)
        return out.view(-1)