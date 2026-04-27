import os
import sys
import argparse
import numpy as np
import torch

# Add parent dir to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module2', 'model')))
from backbone import ResNet18CIFAR
from heads import LinearHead, MLPHead, TemperatureHead

from config import config
from dataset import get_cifar10h_loaders
from metrics import evaluate_all_metrics, print_metrics

def load_model_from_checkpoint(ckpt_path, device='cpu'):
    # Load checkpoint data
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    cfg = checkpoint['config']
    head_variant = cfg['head_variant']
    
    # Setup architecture again
    backbone = ResNet18CIFAR()
    feature_dim = backbone.feature_dim
    
    if head_variant == 'A':
        head = LinearHead(feature_dim, num_classes=10)
    elif head_variant == 'B':
        head = MLPHead(feature_dim, hidden=256, num_classes=10)
    elif head_variant == 'C':
        head = TemperatureHead(feature_dim, num_classes=10)
    else:
        raise ValueError(f"Unknown head type: {head_variant}")
    
    class FullModel(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            z = self.backbone(x)
            return self.head(z)
    
    model = FullModel(backbone, head)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {os.path.basename(ckpt_path)} (Loss: {cfg['loss_name']}, Head: {head_variant})")
    print(f"Best val loss was {checkpoint['val_loss']:.4f} at epoch {checkpoint['epoch']}")
    
    return model, cfg

def run_inference(model, test_loader, device):
    """Run the model on test data and return predictions/targets."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, soft_labels, _, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(soft_labels.numpy())
    
    p_true = np.concatenate(all_targets, axis=0)
    q_pred = np.concatenate(all_preds, axis=0)
    
    return p_true, q_pred

def evaluate_checkpoint(ckpt_path, test_loader, device='cpu'):
    model, cfg = load_model_from_checkpoint(ckpt_path, device)
    p_true, q_pred = run_inference(model, test_loader, device)
    
    metrics = evaluate_all_metrics(p_true, q_pred)
    
    # Store some metadata for the table
    metrics['loss_name'] = cfg['loss_name']
    metrics['head_variant'] = cfg['head_variant']
    metrics['backbone_init'] = cfg.get('backbone_init', 'pretrained')
    
    return metrics

def build_comparison_table(all_metrics):
    """Prints and saves a summary table of all evaluated models."""
    print("\n" + "-"*90)
    print("Model Comparison Summary")
    print("-"*90)
    
    header = (
        f"{'Loss':<10} {'Head':<6} {'Init':<12} "
        f"{'KL↓':<12} {'JSD↓':<12} {'Cosine↑':<12} "
        f"{'SBA↑':<8}"
    )
    print(header)
    print("-" * 90)
    
    for m in all_metrics:
        row = (
            f"{m['loss_name']:<10} "
            f"{m['head_variant']:<6} "
            f"{m['backbone_init']:<12} "
            f"{m['kl_mean']:.4f}±{m['kl_std']:.3f} "
            f"{m['jsd_mean']:.4f}±{m['jsd_std']:.3f} "
            f"{m['cosine_mean']:.4f}±{m['cosine_std']:.3f} "
            f"{m['sba']:.4f}"
        )
        print(row)
    
    print("-" * 90 + "\n")
    
    # Write to text file
    table_path = os.path.join(config['output_dir'], 'results_table.txt')
    with open(table_path, 'w') as f:
        f.write(header + '\n')
        f.write("-" * 90 + '\n')
        for m in all_metrics:
            row = (
                f"{m['loss_name']:<10} "
                f"{m['head_variant']:<6} "
                f"{m['backbone_init']:<12} "
                f"{m['kl_mean']:.4f}±{m['kl_std']:.3f} "
                f"{m['jsd_mean']:.4f}±{m['jsd_std']:.3f} "
                f"{m['cosine_mean']:.4f}±{m['cosine_std']:.3f} "
                f"{m['sba']:.4f}\n"
            )
            f.write(row)
    
    print(f"Saved results table to {table_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint path')
    parser.add_argument('--all', action='store_true', help='Evaluate everything in output_dir')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # Just need the test loader
    _, _, test_loader, _, _, _ = get_cifar10h_loaders(config)
    
    results = []
    
    if args.all:
        # Loop through all checkpoints
        ckpt_dir = config['output_dir']
        files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        print(f"Found {len(files)} checkpoints in {ckpt_dir}")
        
        for f in sorted(files):
            path = os.path.join(ckpt_dir, f)
            m = evaluate_checkpoint(path, test_loader, device)
            print_metrics(m, name=f)
            results.append(m)
    
    elif args.checkpoint:
        m = evaluate_checkpoint(args.checkpoint, test_loader, device)
        print_metrics(m)
        results.append(m)
    
    else:
        print("Please use --checkpoint [path] or --all")
        sys.exit(1)
    
    if len(results) > 1:
        build_comparison_table(results)
    
    print("\nFinished evaluation.")