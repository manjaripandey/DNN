CONFIG = {
    'architecture': 'resnet18_cifar',
    'stem': '3x3_conv_stride1_no_pool',
    'feature_dim': 512,
    'num_classes_pretrain': 10,
    'pretrain_dataset': 'cifar10_train',
    'pretrain_epochs': 200,
    'pretrain_optimizer': 'SGD_nesterov',
    'pretrain_lr': 0.1,
    'pretrain_weight_decay': 5e-4,
    'pretrain_momentum': 0.9,
    'pretrain_batch_size': 128,
    'normalization_mean': [0.4914, 0.4822, 0.4465],
    'normalization_std': [0.2023, 0.1994, 0.2010],
    'random_seed': 42,
    'head_variant': 'A',         # use LinearHead for pretraining
    'lr_schedule': 'cosine',     # 'cosine' or 'step'
    'data_dir': './data',
    'output_dir': '../outputs',
    'num_workers': 4,
    'early_stop_patience': 30,
}
