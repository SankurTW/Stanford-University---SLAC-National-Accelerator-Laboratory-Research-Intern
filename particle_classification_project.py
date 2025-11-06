import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import logging
from tqdm import tqdm
import json
from pathlib import Path
import h5py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParticleJetDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
   
    def __len__(self) -> int:
        return len(self.labels)
   
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
       
        if self.transform:
            x = self.transform(x)
       
        return x, y

class JetAugmentation:
    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        smearing: float = 0.05
    ):
        self.rotation = rotation
        self.flip = flip
        self.smearing = smearing
   
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation and torch.rand(1) > 0.5:
            phi_idx = 2
            rotation_angle = torch.rand(1) * 2 * np.pi
            x[:, phi_idx] = (x[:, phi_idx] + rotation_angle) % (2 * np.pi)
       
        if self.flip and torch.rand(1) > 0.5:
            eta_idx = 1
            x[:, eta_idx] = -x[:, eta_idx]
       
        if self.smearing > 0:
            noise = torch.randn_like(x) * self.smearing * x
            x = x + noise
       
        return x

class ParticleNetSimple(nn.Module):
    def __init__(
        self,
        num_features: int = 4,
        num_classes: int = 2,
        hidden_dim: int = 64
    ):
        super().__init__()
       
        self.input_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
       
        self.edge_conv1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
       
        self.edge_conv2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
       
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
   
    def edge_conv(self, x: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        batch_size, num_particles, features = x.shape
       
        x_i = x.unsqueeze(2).expand(-1, -1, num_particles, -1)
        x_j = x.unsqueeze(1).expand(-1, num_particles, -1, -1)
       
        edge_features = torch.cat([x_i, x_j], dim=-1)
       
        edge_features = edge_features.reshape(-1, edge_features.size(-1))
        edge_features = mlp(edge_features)
        edge_features = edge_features.reshape(batch_size, num_particles, num_particles, -1)
       
        x_updated = edge_features.max(dim=2)[0]
       
        return x_updated
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_particles, _ = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.input_mlp(x)
        x = x.reshape(batch_size, num_particles, -1)
       
        x = self.edge_conv(x, self.edge_conv1)
        x = self.edge_conv(x, self.edge_conv2)
       
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x_global = torch.cat([x_mean, x_max], dim=-1)
       
        out = self.global_mlp(x_global)
       
        return out

class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
       
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
       
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
       
        self.criterion = nn.CrossEntropyLoss()
       
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
       
        self.best_val_loss = float('inf')
        self.patience_counter = 0
   
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
       
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
           
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
           
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
           
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
       
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total
        }
   
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
       
        all_preds = []
        all_targets = []
        all_probs = []
       
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
               
                output = self.model(data)
                loss = self.criterion(output, target)
               
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
               
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
       
        auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
       
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total,
            'auc': auc
        }
   
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = 'checkpoints'
    ):
        Path(checkpoint_dir).mkdir(exist_ok=True)
       
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
       
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
           
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
           
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
           
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )
           
            self.scheduler.step(val_metrics['loss'])
           
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
               
                checkpoint_path = f"{checkpoint_dir}/best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy']
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1
           
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
       
        logger.info("Training complete!")
        return self.history
   
    def plot_training_history(self, save_path: str = 'training_history.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
       
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
       
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
       
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")

def generate_synthetic_jet_data(
    num_samples: int = 10000,
    num_particles_per_jet: int = 30,
    num_classes: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Generating {num_samples} synthetic jet samples...")
   
    features = np.zeros((num_samples, num_particles_per_jet, 4))
    labels = np.random.randint(0, num_classes, num_samples)
   
    for i in range(num_samples):
        if labels[i] == 0:
            pt = np.random.exponential(20, num_particles_per_jet)
            eta = np.random.normal(0, 0.3, num_particles_per_jet)
            phi = np.random.normal(0, 0.3, num_particles_per_jet)
        else:
            pt = np.random.exponential(15, num_particles_per_jet)
            eta = np.random.normal(0, 0.5, num_particles_per_jet)
            phi = np.random.normal(0, 0.5, num_particles_per_jet)
       
        energy = pt * np.cosh(eta)
       
        features[i, :, 0] = pt
        features[i, :, 1] = eta
        features[i, :, 2] = phi
        features[i, :, 3] = energy
   
    return features, labels

def main():
    config = {
        'num_samples': 10000,
        'num_particles': 30,
        'num_classes': 2,
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
   
    logger.info("="*50)
    logger.info("Particle Classification Project for SLAC")
    logger.info("="*50)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
   
    features, labels = generate_synthetic_jet_data(
        num_samples=config['num_samples'],
        num_particles_per_jet=config['num_particles']
    )
   
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
   
    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
   
    train_dataset = ParticleJetDataset(X_train, y_train)
    val_dataset = ParticleJetDataset(X_val, y_val)
   
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
   
    model = ParticleNetSimple(
        num_features=4,
        num_classes=config['num_classes'],
        hidden_dim=64
    )
   
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        lr=config['learning_rate']
    )
   
    history = pipeline.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=10
    )
   
    pipeline.plot_training_history()
   
    logger.info("="*50)
    logger.info("Training complete! Model ready for SLAC research.")
    logger.info("="*50)

if __name__ == "__main__":
    main()