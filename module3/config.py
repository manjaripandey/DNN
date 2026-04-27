# Settings for CIFAR-10H soft-label fine-tuning pipeline

config = {
    # Using the same seed as Module 2 for end-to-end consistency
    'random_seed': 42,

    # File paths
    'data_dir': '../data',
    'cifar10h_probs_file': 'cifar10h-probs.npy',
    'cifar10h_counts_file': 'cifar10h-counts.npy',
    'pretrained_backbone_path': '../module2/outputs/backbone_pretrained.pt',
    'output_dir': './outputs',

    # Dataset split: Splitting the 10,000 CIFAR-10 test images (CIFAR-10H covers these)
    'cifar10h_train_size': 6000,
    'cifar10h_val_size': 2000,
    'cifar10h_test_size': 2000,

    # Normalization (from the resutls of Module 2)
    'normalization_mean': [0.4914, 0.4822, 0.4465],
    'normalization_std': [0.2023, 0.1994, 0.2010],

    # Model settings
    'architecture': 'resnet18_cifar',
    'feature_dim': 512,
    'num_classes': 10,
    'primary_head': 'A', # Head variant 'A' used by default
    'all_heads': ['A', 'B', 'C'],

    # Training hyperparams
    'optimizer': 'AdamW',
    'lr': 1e-4, # Lower LR for fine-tuning
    'weight_decay': 1e-4,
    'lr_schedule': 'cosine',
    'finetune_epochs': 60,
    'early_stop_patience': 15,
    'batch_size': 128,
    'num_workers': 2,

    # Data Augmentation
    'augmentation': ['random_horizontal_flip', 'random_crop_pad4'],

    # Loss Functions
    'loss_functions': ['KL', 'JSD', 'SoftCE', 'Custom'],
    'lambda_entropy': 1.0, # entropy-error penalty weight
    'emd_distance_type': 'semantic',

    # Metrics and ablation
    'precision_at_k': [100, 200, 500],
    'backbone_init_variants': ['pretrained', 'random'],

    # Logging
    'checkpoint_pattern': 'checkpoint_{loss}_{head}.pt',
    'log_every_n_batches': 20,
}
