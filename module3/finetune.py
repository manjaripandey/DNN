import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Path setup to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module2', 'model')))
from backbone import ResNet18CIFAR
from heads import LinearHead, MLPHead, TemperatureHead

from config import config
from dataset import get_cifar10h_loaders
from losses import get_loss_fn

def set_seed(seed):
    """Ensure some level of reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(head_variant, backbone_init, cfg):
    """
    Constructs the model based on the requested head and backbone init.
    """
    backbone = ResNet18CIFAR()

    # Initializing backbone
    if backbone_init == 'pretrained':
        ckpt_path = cfg['pretrained_backbone_path']
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing backbone weights at {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        print(f"Loaded pretrained backbone from {os.path.basename(ckpt_path)}")
    else:
        print("Using random backbone (no pretraining)")

    # Choosing the head
    feature_dim = backbone.feature_dim
    if head_variant == 'A':
        head = LinearHead(feature_dim, num_classes=10)
    elif head_variant == 'B':
        head = MLPHead(feature_dim, hidden=256, num_classes=10)
    elif head_variant == 'C':
        head = TemperatureHead(feature_dim, num_classes=10)
    else:
        raise ValueError(f"Unknown head choice: {head_variant}")

    # Wrap them together
    class FullModel(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            z = self.backbone(x)
            return self.head(z)
    
    model = FullModel(backbone, head)
    return model

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, soft_labels, _, _ in loader:
        images, soft_labels = images.to(device), soft_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = loss_fn(outputs, soft_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, soft_labels, _, _ in loader:
            images, soft_labels = images.to(device), soft_labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, soft_labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def start_finetuning(loss_name, head_variant, backbone_init, cfg):
    """Main training orchestrator for a single configuration."""
    set_seed(cfg['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n")
    print(f"Starting experiment: {loss_name} | {head_variant} | {backbone_init}")
    print(f"Training on: {device}")
    print("\n")
    
    train_loader, val_loader, _, _, _, _ = get_cifar10h_loaders(cfg)
    model = build_model(head_variant, backbone_init, cfg).to(device)
    loss_fn = get_loss_fn(loss_name, cfg)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['finetune_epochs'])
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, cfg['finetune_epochs'] + 1):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        
        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            
            ckpt_name = f"checkpoint_{loss_name}_{head_variant}_{backbone_init}.pt"
            ckpt_path = os.path.join(cfg['output_dir'], ckpt_name)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'loss_name': loss_name,
                    'head_variant': head_variant,
                    'backbone_init': backbone_init,
                    'random_seed': cfg['random_seed'],
                }
            }, ckpt_path)
            # print(f"  Saved best model.")
        else:
            patience_counter += 1
        
        if patience_counter >= cfg['early_stop_patience']:
            print(f"Stopping early after {cfg['early_stop_patience']} epochs of no improvement.")
            break
    
    print(f"Best validation loss for this run: {best_loss:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, required=True, choices=['KL', 'JSD', 'SoftCE', 'Custom', 'EMD'])
    parser.add_argument('--head', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--init', type=str, default='pretrained', choices=['pretrained', 'random'])
    args = parser.parse_args()
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    start_finetuning(
        loss_name=args.loss,
        head_variant=args.head,
        backbone_init=args.init,
        cfg=config
    )
    
    print("Fine-tuning finished.")