# Module 2 — Model Architecture & Pretraining

## Quick start

```bash
# From the DNNproject root:
python -m module2.pretrain.pretrain
```

Requires: `torch`, `torchvision`, `matplotlib`.  
CIFAR-10 data is downloaded automatically to `module2/pretrain/data/`.

---

## Loading `backbone_pretrained.pt` (for Module 3)

```python
import torch
from module2.model import ResNet18CIFAR, LinearHead, MLPHead

checkpoint = torch.load('module2/outputs/backbone_pretrained.pt', map_location='cpu')
config = checkpoint['config']

model = ResNet18CIFAR()
model.load_state_dict(checkpoint['backbone_state_dict'])

# Attach head of choice (re-initialized — NOT from checkpoint):
# head = LinearHead(feature_dim=512, num_classes=10)   # Head A
# head = MLPHead(feature_dim=512, hidden=256, num_classes=10)  # Head B

# Use these normalization stats for ALL downstream data loading:
mean = config['normalization_mean']   # [0.4914, 0.4822, 0.4465]
std  = config['normalization_std']    # [0.2023, 0.1994, 0.2010]
```

---

## Parameter counts

| Component       | Architecture                    | Parameters  |
|-----------------|---------------------------------|-------------|
| Backbone        | ResNet-18 CIFAR-adapted         | 11,168,832  |
| Head Variant A  | Linear(512→10) + Softmax        | 5,130       |
| Head Variant B  | MLP(512→256→10) + BN + Dropout  | 134,410     |
| Full Model A    | Backbone + Head A               | 11,173,962  |
| Full Model B    | Backbone + Head B               | 11,303,242  |

---

## Outputs

| File                              | Description                          |
|-----------------------------------|--------------------------------------|
| `outputs/backbone_pretrained.pt`  | Backbone checkpoint (no head)        |
| `outputs/pretrain_loss_curve.png` | Training loss per epoch (200 epochs) |
| `outputs/pretrain_acc_curve.png`  | Val accuracy per epoch               |
| `figures/architecture_diagram.png`| Architecture diagram for report      |

---

## Sanity checks before handoff

- [ ] Val accuracy ≥ 90 % (target ~93–95 %)
- [ ] `torch.load` on fresh session — no errors
- [ ] `backbone_state_dict` keys match `backbone.py`
- [ ] `config['normalization_mean/std']` present in checkpoint
- [ ] Pretraining head NOT in checkpoint
- [ ] Curves saved at ≥ 150 DPI
- [ ] Parameter table verified with `sum(p.numel() for p in model.parameters())`
