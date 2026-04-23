"""Main pretraining script for Module 2.

Pretrain ResNet-18 CIFAR backbone on CIFAR-10 hard labels (50k images).
Saves backbone_pretrained.pt (backbone weights only, no head) to outputs/.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from module2.model import FullModel
from module2.pretrain.config import CONFIG
from module2.pretrain.train_utils import train_epoch, eval_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(cfg):
    mean, std = cfg['normalization_mean'], cfg['normalization_std']
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(cfg['data_dir'], train=True, download=True, transform=train_transforms)
    val_set = datasets.CIFAR10(cfg['data_dir'], train=False, download=True, transform=val_transforms)
    train_loader = DataLoader(train_set, batch_size=cfg['pretrain_batch_size'],
                              shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)
    return train_loader, val_loader


def save_curves(train_losses, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Pretraining Loss Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pretrain_loss_curve.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, [a * 100 for a in val_accs])
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Pretraining Validation Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pretrain_acc_curve.png'), dpi=150)
    plt.close()


def main():
    cfg = CONFIG
    set_seed(cfg['random_seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(cfg['output_dir'], exist_ok=True)

    train_loader, val_loader = get_loaders(cfg)

    model = FullModel(head_variant=cfg['head_variant']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['pretrain_lr'],
        momentum=cfg['pretrain_momentum'],
        weight_decay=cfg['pretrain_weight_decay'],
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['pretrain_epochs'])

    train_losses, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_patience = cfg.get('early_stop_patience', 30)
    epochs_no_improve = 0

    for epoch in range(1, cfg['pretrain_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = eval_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch:3d}/{cfg["pretrain_epochs"]}] '
              f'Loss: {train_loss:.4f}  Val Acc: {val_acc*100:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            checkpoint = {
                'backbone_state_dict': model.backbone.state_dict(),
                'epoch': epoch,
                'val_accuracy': best_val_acc,
                'config': {
                    'architecture': cfg['architecture'],
                    'stem': cfg['stem'],
                    'feature_dim': cfg['feature_dim'],
                    'num_classes_pretrain': cfg['num_classes_pretrain'],
                    'pretrain_dataset': cfg['pretrain_dataset'],
                    'pretrain_epochs': cfg['pretrain_epochs'],
                    'pretrain_optimizer': cfg['pretrain_optimizer'],
                    'pretrain_lr': cfg['pretrain_lr'],
                    'pretrain_weight_decay': cfg['pretrain_weight_decay'],
                    'pretrain_batch_size': cfg['pretrain_batch_size'],
                    'normalization_mean': cfg['normalization_mean'],
                    'normalization_std': cfg['normalization_std'],
                    'random_seed': cfg['random_seed'],
                },
            }
            torch.save(checkpoint, os.path.join(cfg['output_dir'], 'backbone_pretrained.pt'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f'\nEarly stopping at epoch {epoch} '
                      f'(no improvement for {early_stop_patience} epochs)')
                break

    print(f'\nBest val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}')
    save_curves(train_losses, val_accs, cfg['output_dir'])
    print('Curves saved. Done.')


if __name__ == '__main__':
    main()
