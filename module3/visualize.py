import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module2', 'model')))
from backbone import ResNet18CIFAR
from heads import LinearHead, MLPHead, TemperatureHead

from config import config
from dataset import get_cifar10h_loaders
from metrics import compute_entropy
from evaluate import load_model_from_checkpoint, run_inference

def plot_entropy_scatter(p_true, q_pred, save_path, model_name="Model"):
    """Creates a scatter plot comparing human vs model entropy."""
    h_true = compute_entropy(p_true)
    h_pred = compute_entropy(q_pred)
    
    from scipy.stats import pearsonr
    r, _ = pearsonr(h_true, h_pred)
    
    plt.figure(figsize=(7, 7))
    plt.scatter(h_true, h_pred, alpha=0.3, s=12, c='royalblue')
    
    # Perfect prediction line
    limit = max(h_true.max(), h_pred.max())
    plt.plot([0, limit], [0, limit], 'r--', alpha=0.6, label='Ideal')
    
    plt.xlabel('Human Entropy H(p)')
    plt.ylabel('Model Entropy H(q)')
    plt.title(f'{model_name}\nPearson Correlation: {r:.4f}')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved entropy scatter to {save_path}")

def plot_loss_curves(train_losses, val_losses, save_path, model_name="Model"):
    """Plots training and validation loss."""
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label='Train', lw=1.5)
    plt.plot(val_losses, label='Val', lw=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Progress')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curves to {save_path}")

def plot_metric_comparison(all_metrics, metric_key, save_path):
    """Bar chart comparing a specific metric across all runs."""
    labels = [f"{m['loss_name']}\n{m['head_variant']}" for m in all_metrics]
    vals = [m[metric_key] for m in all_metrics]
    
    plt.figure(figsize=(10, 5))
    colors = ['gray'] * len(vals)
    
    # Highlight the best run
    higher_better = any(x in metric_key for x in ['precision', 'sba', 'cosine', 'pearson'])
    best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
    colors[best_idx] = 'darkorange'
    
    plt.bar(labels, vals, color=colors, alpha=0.8)
    plt.title(f'Comparison: {metric_key}')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

def plot_qualitative_grid(p_true, q_pred, images, save_path):
    """Shows a grid of images with low and high entropy examples."""
    h_true = compute_entropy(p_true)
    sorted_idx = np.argsort(h_true)
    
    # Pick 12 samples across the entropy spectrum
    indices = [sorted_idx[i] for i in np.linspace(0, len(sorted_idx)-1, 12, dtype=int)]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = images[idx].transpose(1, 2, 0)
        # Simple denormalization
        img = (img * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"H_true: {h_true[idx]:.2f}\nH_pred: {compute_entropy(q_pred[idx:idx+1])[0]:.2f}", fontsize=9)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved qualitative grid to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint for single model plots')
    parser.add_argument('--all', action='store_true', help='Plot for all checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader, _, _, split_indices = get_cifar10h_loaders(config)
    test_idx = split_indices['test']
    
    # Need original images for visualization
    from torchvision import datasets
    raw_data = datasets.CIFAR10(config['data_dir'], train=False, download=True)
    
    if args.checkpoint:
        model, cfg = load_model_from_checkpoint(args.checkpoint, device)
        p_true, q_pred = run_inference(model, test_loader, device)
        
        name = os.path.basename(args.checkpoint).replace('.pt', '')
        out_dir = config['output_dir']
        
        plot_entropy_scatter(p_true, q_pred, os.path.join(out_dir, f'{name}_scatter.png'), name)
        
        # Prepare test images
        imgs = np.array([np.array(raw_data[i][0]) / 255.0 for i in test_idx]).transpose(0, 3, 1, 2)
        plot_qualitative_grid(p_true, q_pred, imgs, os.path.join(out_dir, f'{name}_qualitative.png'))
        
    elif args.all:
        ckpt_dir = config['output_dir']
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        for ckpt in sorted(ckpts):
            path = os.path.join(ckpt_dir, ckpt)
            model, _ = load_model_from_checkpoint(path, device)
            p_true, q_pred = run_inference(model, test_loader, device)
            
            name = ckpt.replace('.pt', '')
            plot_entropy_scatter(p_true, q_pred, os.path.join(ckpt_dir, f'{name}_scatter.png'), name)
            
    else:
        print("Use --checkpoint or --all")
        sys.exit(1)