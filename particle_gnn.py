import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Optional, Tuple, List
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
   
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        edge_features = self.mlp(edge_features)
       
        out = torch.zeros_like(x[:, :edge_features.size(-1)])
        out.index_add_(0, row, edge_features)
       
        return out

class ParticleAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.gat = GATConv(in_channels, in_channels // heads, heads=heads,
                          Dropout=0.2, concat=True)
        self.norm = nn.LayerNorm(in_channels)
   
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.gat(x, edge_index)
        x = self.norm(x + identity)
        return x

class AdvancedParticleGNN(nn.Module):
    def __init__(
        self,
        num_features: int = 4,
        num_classes: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_edge_features: bool = True
    ):
        super().__init__()
       
        self.num_features = num_features
        self.num_classes = num_classes
       
        self.input_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
       
        self.edge_convs = nn.ModuleList([
            EdgeConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers // 2)
        ])
       
        self.attention_layers = nn.ModuleList([
            ParticleAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers // 2)
        ])
       
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
       
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
       
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
       
        self.energy_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
       
        self.dropout = nn.Dropout(dropout)
   
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_encoder(x)
       
        layer_idx = 0
        for edge_conv, attention in zip(self.edge_convs, self.attention_layers):
            identity = x
            x = edge_conv(x, edge_index)
            x = self.layer_norms[layer_idx](x + identity)
            x = self.dropout(x)
            layer_idx += 1
           
            identity = x
            x = attention(x, edge_index)
            x = self.layer_norms[layer_idx](x + identity)
            x = self.dropout(x)
            layer_idx += 1
       
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=-1)
        x_global = self.global_pool(x_global)
       
        class_logits = self.classifier(x_global)
        energy = self.energy_regressor(x_global)
       
        return class_logits, energy
   
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_encoder(x)
       
        for edge_conv, attention in zip(self.edge_convs, self.attention_layers):
            x = edge_conv(x, edge_index)
            x = attention(x, edge_index)
       
        return global_mean_pool(x, batch)

class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
   
    def forward(
        self,
        class_logits: torch.Tensor,
        energy_pred: torch.Tensor,
        class_labels: torch.Tensor,
        energy_true: torch.Tensor,
        total_energy: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        loss_class = self.ce_loss(class_logits, class_labels)
       
        loss_energy = self.mse_loss(energy_pred.squeeze(), energy_true)
       
        loss_physics = torch.tensor(0.0, device=class_logits.device)
        if total_energy is not None:
            predicted_total = energy_pred.sum()
            loss_physics = torch.abs(predicted_total - total_energy.sum()) / total_energy.sum()
       
        total_loss = (
            self.alpha * loss_class +
            self.beta * loss_energy +
            self.gamma * loss_physics
        )
       
        loss_dict = {
            'total': total_loss.item(),
            'classification': loss_class.item(),
            'energy': loss_energy.item(),
            'physics': loss_physics.item()
        }
       
        return total_loss, loss_dict

class ParticleDataPreprocessor:
    def __init__(
        self,
        k_neighbors: int = 8,
        radius: float = 0.4,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None
    ):
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.feature_mean = feature_mean
        self.feature_std = feature_std
   
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None:
            self.feature_mean = features.mean(axis=0)
            self.feature_std = features.std(axis=0) + 1e-6
       
        return (features - self.feature_mean) / self.feature_std
   
    def construct_graph(
        self,
        features: np.ndarray,
        use_radius: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from scipy.spatial import cKDTree
       
        spatial_coords = features[:, :2]
        tree = cKDTree(spatial_coords)
       
        edges = []
        for i in range(len(features)):
            if use_radius:
                neighbors = tree.query_ball_point(spatial_coords[i], self.radius)
            else:
                _, neighbors = tree.query(spatial_coords[i], k=self.k_neighbors + 1)
                neighbors = neighbors[1:]
           
            for j in neighbors:
                if i != j:
                    edges.append([i, j])
       
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
       
        return edge_index
   
    def create_pytorch_geometric_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        energy: np.ndarray
    ) -> Data:
        features_norm = self.normalize_features(features)
        edge_index = self.construct_graph(features_norm)
       
        data = Data(
            x=torch.tensor(features_norm, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            energy=torch.tensor(energy, dtype=torch.float)
        )
       
        return data

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    device: str = 'cuda'
) -> dict:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
   
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
       
        class_logits, energy_pred = model(batch.x, batch.edge_index, batch.batch)
       
        loss, loss_dict = criterion(
            class_logits, energy_pred,
            batch.y, batch.energy
        )
       
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
       
        total_loss += loss.item()
        pred = class_logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
   
    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total
    }

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    device: str = 'cuda'
) -> dict:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
           
            class_logits, energy_pred = model(batch.x, batch.edge_index, batch.batch)
            loss, _ = criterion(class_logits, energy_pred, batch.y, batch.energy)
           
            total_loss += loss.item()
            pred = class_logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
   
    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total
    }

if __name__ == "__main__":
    logger.info("Initializing Advanced Particle GNN for SLAC research...")
   
    model = AdvancedParticleGNN(
        num_features=4,
        num_classes=5,
        hidden_dim=128,
        num_layers=6,
        num_heads=8
    )
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
   
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {device}")
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysicsInformedLoss(alpha=1.0, beta=0.3, gamma=0.1)
   
    logger.info("Model ready for training on SLAC particle physics data!")