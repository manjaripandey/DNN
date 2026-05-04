# Module 3 — Fine-tuning & Evaluation

**Team Member 3's deliverable:** Fine-tune pretrained backbone on CIFAR-10H soft labels, evaluate with 7 metrics, generate comparison tables and visualizations.

---

## Quick Start

```bash
# From project root:
cd module3

# Run the full pipeline (fine-tune → evaluate → visualize)
python main.py

# Or run components individually:
python finetune.py --loss KL --head A
python evaluate.py --all
python visualize.py --all
```

---

## File Structure

```
module3/
├── config.py          # All hyperparameters and paths
├── dataset.py         # CIFAR-10H data loader with train/val/test split
├── losses.py          # KL, JSD, SoftCE, Custom, EMD loss functions
├── finetune.py        # Training loop for soft-label fine-tuning
├── metrics.py         # All 7 evaluation metrics
├── evaluate.py        # Run inference + compute metrics + build table
├── visualize.py       # Generate all required plots
├── main.py            # Master orchestration script
└── outputs/           # All results saved here
    ├── checkpoint_*.pt
    ├── results_table.txt
    └── *.png
```

---

## What Each File Does

### `config.py`
Single source of truth for every setting:
- Data paths (CIFAR-10H files, pretrained backbone)
- Train/val/test split sizes (6000/2000/2000)
- Fine-tuning hyperparameters (lr=1e-4, AdamW, cosine schedule)
- Loss function list and custom loss hyperparameters
- Augmentation policy (flip + crop, no semantic changes)

### `dataset.py`
- Loads `cifar10h-probs.npy` (10,000 × 10 soft labels)
- Aligns with CIFAR-10 test images (guaranteed by CIFAR-10H authors)
- Splits into reproducible train/val/test sets (seed=42)
- Returns DataLoaders yielding `(image, soft_label, hard_label, index)`
- Runs sanity checks: row sums = 1, no NaN/Inf, entropy range

### `losses.py`
Five loss functions for soft-label distribution matching:
1. **KL Divergence** — primary baseline, asymmetric
2. **JSD** — symmetric, bounded variant of KL
3. **Soft Cross-Entropy** — equivalent to KL + constant
4. **Custom** — KL + entropy penalty: `KL(p||q) + λ*(H(p) - H(q))²`
5. **EMD** (bonus) — Wasserstein distance with semantic class-distance matrix

All handle numerical stability (clipping before log).

### `finetune.py`
Training pipeline:
- Loads pretrained backbone from Module 2
- Attaches fresh head (A/B/C)
- Fine-tunes on 6,000 soft-label images
- Uses AdamW optimizer + cosine LR schedule
- Early stopping (patience=15 epochs)
- Saves best checkpoint based on val loss

Supports:
- `--loss {KL, JSD, SoftCE, Custom, EMD}`
- `--head {A, B, C}`
- `--init {pretrained, random}` (for ablation)

### `metrics.py`
Implements all 7 required metrics:
1. **KL Divergence** (mean ± std)
2. **JSD** (mean ± std)
3. **EMD** (mean ± std, using semantic distance matrix)
4. **Cosine Similarity** (mean ± std)
5. **Entropy Correlation** (Pearson r/p, Spearman r/p)
6. **Precision@K** (K=100, 200, 500) — do predicted high-entropy rankings match true rankings?
7. **SBA** (Soft-label Balanced Accuracy) — class-weighted accuracy

### `evaluate.py`
- Loads trained checkpoints
- Runs inference on 2,000-sample test set
- Computes all 7 metrics
- Builds comparison table (rows=models, cols=metrics)
- Saves to `results_table.txt`

### `visualize.py`
Generates all required plots:
- **Entropy scatter:** predicted vs true entropy (with Pearson r)
- **Qualitative grid:** 15 examples spanning low→high disagreement
- Optional: loss curves, metric comparison bars

### `main.py`
Orchestrates the full pipeline:
1. Fine-tunes all (loss × head) combinations
2. Evaluates all checkpoints
3. Generates all visualizations

Flags:
- `--losses KL Custom` — run subset
- `--heads A B` — run subset
- `--skip-finetune` / `--skip-eval` / `--skip-viz`

---

## Ablations Covered

The code implements these ablations (required: pick any 3):

| ID | Ablation | Implementation |
|----|----------|----------------|
| **B** | Loss Comparison | Compare KL vs JSD vs SoftCE vs Custom (primary experiment) |
| **D** | Head Architecture | Compare Head A vs B vs C |
| **C** | Training Strategy | `--init pretrained` vs `--init random` |
| **A** | Backbone Init | Pretrained (Module 2) vs random |

Ablations B and D run automatically. C requires one extra finetune call with `--init random`.

---

## Outputs

After running `python main.py`, you'll have:

```
outputs/
├── checkpoint_KL_A_pretrained.pt
├── checkpoint_JSD_A_pretrained.pt
├── checkpoint_Custom_A_pretrained.pt
├── checkpoint_KL_A_pretrained.pt
├── checkpoint_KL_A_pretrained.pt
├── checkpoint_KL_A_random.pt
├── results_table.txt                   # ← Comparison table for report
├── checkpoint_KL_A_pretrained_entropy_scatter.png
├── checkpoint_KL_A_pretrained_qualitative.png
└── ... (plots for each checkpoint)
```

---

## Expected Results

**Hypothesis:**
- **KL** will be strong but may overfit to high-confidence regions
- **JSD** will be more robust, better generalization
- **Custom** (KL + entropy penalty) should win on entropy prediction quality (Pearson/Spearman correlation)
- **Head C** (TemperatureHead) may naturally produce softer distributions → better soft-label fit
- **Pretrained backbone >> random** (ablation C will confirm this)

**Sanity checks:**
- Val accuracy from Module 2 ≥ 90% (confirms backbone quality)
- Entropy range: [0, ~3.3] (0 = full agreement, 3.3 = uniform)
- KL/JSD should decrease over epochs
- Pearson r (entropy) > 0.5 for a decent model

---

## Common Issues & Fixes

**"FileNotFoundError: backbone_pretrained.pt"**
→ Run Module 2 first: `python module2/pretrain.py`

**"AssertionError: soft labels don't sum to 1"**
→ Check CIFAR-10H file version. Should be normalized probabilities, not raw counts.

**Loss = NaN during training**
→ Clipping is already in losses.py. Check learning rate (too high?).

**Low Pearson correlation (<0.3)**
→ Model is matching distributions but not entropy levels. Try the Custom loss (entropy penalty).

**All metrics identical across loss functions**
→ Check that you're actually using different loss functions. Print `loss_name` during training.

---

## License & Attribution

This module fine-tunes the ResNet-18 CIFAR backbone from Module 2 on the CIFAR-10H dataset:

**CIFAR-10H:**
- Paper: "Human uncertainty makes classification more robust" (Peterson et al., ICCV 2019)
- Data: 10,000 CIFAR-10 test images × ~50 human annotations each
- URL: https://github.com/jcpeterson/cifar-10h

**Module dependencies:**
- Module 1: data loading, entropy visualization
- Module 2: pretrained backbone (`backbone_pretrained.pt`)