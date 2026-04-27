import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import config

cfg = config.config

# Dataset class to wrap CIFAR-10 images with CIFAR-10H soft labels
class CIFAR10HDataset(Dataset):
    def __init__(self, cifar10_test, soft_labels, indices, transform=None):
        self.cifar10_test = cifar10_test
        self.soft_labels = torch.tensor(soft_labels, dtype=torch.float32)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # global index in the 10k set
        global_idx = self.indices[i]
        image, hard_label = self.cifar10_test[global_idx]

        if self.transform:
            image = self.transform(image)

        soft_label = self.soft_labels[global_idx]
        return image, soft_label, hard_label, global_idx

def run_sanity_checks(soft_labels, counts):
    """ checks for data integrity """
    print("\nRunning CIFAR-10H Sanity Checks...")

    # check shapes
    assert soft_labels.shape == (10000, 10)
    assert counts.shape == (10000, 10)
    print(f"  Shape check passed: {soft_labels.shape}")

    # check row sums
    row_sums = soft_labels.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    print("  Row sums check passed.")

    # check for nans
    assert not np.any(np.isnan(soft_labels))
    assert not np.any(np.isinf(soft_labels))
    print("  No NaN/Inf values.")

    # counts check
    assert np.all(counts >= 0)
    print("  Counts are non-negative.")

    # calculate entropy
    eps = 1e-10
    p = np.clip(soft_labels, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=1)
    
    print(f"  Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    print(f"  Perfect agreement count: {np.sum(entropy < 0.01)}")
    print(f"  Strong disagreement count: {np.sum(entropy > 2.0)}")
    print("Sanity checks done.\n")

    return entropy

def get_transforms(cfg):
    mean = cfg['normalization_mean']
    std = cfg['normalization_std']

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, eval_transform

def get_cifar10h_loaders(cfg):
    # load numpy files
    probs_path = os.path.join(cfg['data_dir'], cfg['cifar10h_probs_file'])
    counts_path = os.path.join(cfg['data_dir'], cfg['cifar10h_counts_file'])

    soft_labels = np.load(probs_path).astype(np.float32)
    counts = np.load(counts_path).astype(np.float32)

    # run checks
    entropy = run_sanity_checks(soft_labels, counts)

    # load raw cifar10 test set
    cifar10_test_raw = datasets.CIFAR10(
        root=cfg['data_dir'],
        train=False,
        download=True,
        transform=None
    )

    # create splits using fixed seed
    rng = np.random.default_rng(cfg['random_seed'])
    all_indices = np.arange(10000)
    rng.shuffle(all_indices)

    n_train = cfg['cifar10h_train_size']
    n_val = cfg['cifar10h_val_size']

    train_idx = all_indices[:n_train].tolist()
    val_idx = all_indices[n_train : n_train + n_val].tolist()
    test_idx = all_indices[n_train + n_val :].tolist()

    print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_transform, eval_transform = get_transforms(cfg)

    # create datasets
    train_ds = CIFAR10HDataset(cifar10_test_raw, soft_labels, train_idx, transform=train_transform)
    val_ds = CIFAR10HDataset(cifar10_test_raw, soft_labels, val_idx, transform=eval_transform)
    test_ds = CIFAR10HDataset(cifar10_test_raw, soft_labels, test_idx, transform=eval_transform)

    # dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, entropy, soft_labels, {'train': train_idx, 'val': val_idx, 'test': test_idx}

if __name__ == '__main__':
    # local test
    train_loader, val_loader, test_loader, entropy, labels, splits = get_cifar10h_loaders(cfg)
    
    img, sl, hl, idx = next(iter(train_loader))
    print(f"Batch loaded. Image shape: {img.shape}")
