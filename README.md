# CIFAR-10H Soft-Label Learning

Fine-tuning a ResNet-18 backbone on CIFAR-10H to match human annotation distributions using soft-label loss functions, with Grad-CAM interpretability analysis.

## Overview

Standard image classification discards annotator uncertainty by collapsing votes into a single hard label. This project leverages **CIFAR-10H** (Peterson et al., 2019), which provides ~50 human annotations per CIFAR-10 test image as a probability distribution, to train models that better capture human perceptual uncertainty.

## Setup

```bash
git clone https://github.com/manjaripandey/DNN.git
cd DNN
pip install -r requirements.txt
```

**Requirements:** `torch`, `torchvision`, `numpy`, `matplotlib`, `scipy`

## Usage

### 1. Pretrain on CIFAR-10 (hard labels)
```bash
python train_pretrain.py --epochs 300 --lr 0.1 --seed 42
```

### 2. Fine-tune on CIFAR-10H (soft labels)
```bash
python train_finetune.py --loss kl --init pretrained
python train_finetune.py --loss jsd --init pretrained
python train_finetune.py --loss softce --init pretrained
python train_finetune.py --loss custom --init pretrained
python train_finetune.py --loss kl --init random   # baseline
```

### 3. Evaluate
```bash
python evaluate.py --checkpoint checkpoints/checkpoint_KL_A_pretrained.pt
```

### 4. Generate Grad-CAM visualisations
```bash
python gradcam.py --checkpoint checkpoints/checkpoint_KL_A_pretrained.pt \
                  --mode high_entropy   # or low_entropy / disagreement
```

## Model Architecture

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-18, CIFAR-adapted (3×3 stem, no max-pool) |
| Features | 512-d global average pooled vector |
| Head A | Linear(512→10) + Softmax |
| Head B | Linear→BN→ReLU→Dropout(0.3)→Linear + Softmax |
| Head C | Linear(512→10) / T + Softmax, T learnable |

All fine-tuning experiments use **Head A** for comparability.

## Loss Functions

| Loss | Description |
|------|-------------|
| `kl` | KL Divergence — minimises per-sample information-theoretic distance |
| `jsd` | Jensen-Shannon Divergence — symmetric, bounded [0,1], numerically stable |
| `softce` | Soft Cross-Entropy — treats soft labels as target probability vectors |
| `custom` | Cross-entropy + entropy-error penalty (λ=1.0) |

## Pretraining Configuration

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 train (50,000 images) |
| Epochs | 200 (early-stop patience 30) |
| Optimiser | SGD Nesterov, lr=0.1, momentum=0.9, wd=0.0005 |
| LR Schedule | Cosine annealing |
| Batch size | 128 |
| Seed | 42 |

## Evaluation Metrics

- **KL Divergence** ↓ — primary distribution-matching metric
- **JSD** ↓ — symmetric divergence
- **Cosine Similarity** ↑ — directional alignment of distributions  
- **SBA** ↑ — soft balanced accuracy

## Citation

```bibtex
@inproceedings{peterson2019human,
  title={Human uncertainty makes classification more robust},
  author={Peterson, Joshua C and Battleday, Ruairidh M and Griffiths, Thomas L and Russakovsky, Olga},
  booktitle={ICCV},
  year={2019}
}

@inproceedings{selvaraju2017grad,
  title={Grad-CAM: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasath R and others},
  booktitle={ICCV},
  year={2017}
}
```
