import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# ────────────────────────────
# Spatial Graph Convolution
# ────────────────────────────
class SpatialGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_dim)

    def forward(self, x, edge_index):
        B, T, N, F = x.shape
        x = x.view(-1, N, F).permute(0, 2, 1)  # [B*T, F, N]

        outputs = []
        for b in range(B):
            temp = []
            for t in range(T):
                x_bt = x[b * T + t].permute(1, 0)  # [N, F]
                x_gcn = self.gcn(x_bt, edge_index)  # [N, hidden]
                temp.append(x_gcn)
            outputs.append(torch.stack(temp, dim=0))  # [T, N, hidden]

        return torch.stack(outputs, dim=0)  # [B, T, N, hidden]

# ────────────────────────────
# Spatial Attention
# ────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.proj(x)  # [B, T, N, 1]
        weights = torch.softmax(scores, dim=2)  # across nodes
        return x * weights

# ────────────────────────────
# Temporal Modeling
# ────────────────────────────
class TemporalGRU(nn.Module):
    def __init__(self, hidden_dim, out_steps):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_steps)

    def forward(self, x):
        B, T, N, H = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, H)
        out, _ = self.gru(x)
        out = self.proj(out[:, -1])
        return out.view(B, N, -1).permute(0, 2, 1).unsqueeze(-1)

# ────────────────────────────
# Temporal Attention
# ────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.proj(x)  # [B, T, N, 1]
        weights = torch.softmax(scores, dim=1)  # across time
        return x * weights

# ────────────────────────────
# Final ReX-GNN Model
# ────────────────────────────
class ReXGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_steps, num_nodes):
        super().__init__()
        self.spatial_gcn = SpatialGCN(in_channels, hidden_dim)
        self.spatial_attn = SpatialAttention(hidden_dim)
        self.temporal_gru = TemporalGRU(hidden_dim, out_steps)
        self.temporal_attn = TemporalAttention(hidden_dim)

    def forward(self, x, edge_index):
        x = x.squeeze(-1).unsqueeze(-1)  # [B, T, N, 1]
        h_spatial = self.spatial_gcn(x, edge_index)
        h_spatial = self.spatial_attn(h_spatial)
        h_temporal = self.temporal_attn(h_spatial)
        return self.temporal_gru(h_temporal)
