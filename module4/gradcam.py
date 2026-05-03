"""
Module 4 — Grad-CAM Visualisation
Runs Grad-CAM on the KL and JSD fine-tuned checkpoints across 24 carefully
selected test images (8 high-entropy, 8 low-entropy, 8 disagreement cases).

Hooks are registered directly on the model — no grad-cam library used.
All hooks are removed after use to avoid memory leaks.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms

# ── path setup so module3 imports work ───────────────────────────────────────
MODULE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module3'))
MODULE2_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module2', 'model'))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for p in [MODULE3_DIR, MODULE2_MODEL_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from module2.model.backbone import ResNet18CIFAR
from module2.model.heads import LinearHead
# from module3.dataset import get_cifar10_loaders
from module3.config import config as cfg       # module3/config.py

# ── constants ─────────────────────────────────────────────────────────────────
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

DENORM_MEAN = np.array([0.4914, 0.4822, 0.4465])
DENORM_STD  = np.array([0.2023, 0.1994, 0.2010])

OUT_KL  = os.path.join(os.path.dirname(__file__), 'figures', 'gradcam')
OUT_JSD = os.path.join(os.path.dirname(__file__), 'figures', 'gradcam_JSD')

os.makedirs(OUT_KL,  exist_ok=True)
os.makedirs(OUT_JSD, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Model loading
# ══════════════════════════════════════════════════════════════════════════════

class FullModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


def load_model(ckpt_path: str, device: torch.device) -> tuple[FullModel, dict]:
    """Reconstruct architecture from checkpoint metadata and load weights."""
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt['config']
    head_variant = saved_cfg.get('head_variant', 'A')

    backbone = ResNet18CIFAR()

    if head_variant == 'A':
        head = LinearHead(backbone.feature_dim, num_classes=10)
    else:
        raise ValueError(f"Module 4 only supports head A; got {head_variant}")

    model = FullModel(backbone, head)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    print(f"  Loaded: {os.path.basename(ckpt_path)}  "
          f"(loss={saved_cfg['loss_name']}, "
          f"epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, saved_cfg


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Grad-CAM core  (hook-based, no external library)
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Registers forward + backward hooks on `target_layer`.
    Call .compute(input_tensor, class_idx) -> heatmap (H×W numpy float32 in [0,1]).
    Call .remove() when done to free hooks.
    """

    def __init__(self, model: FullModel, target_layer: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def compute(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        x           : (1, C, H, W) tensor already on the correct device
        class_idx   : class to visualise (int)
        returns     : (H, W) heatmap, values in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(x)                      # forward pass → hooks fire
        score = logits[0, class_idx]
        score.backward()                            # backward pass → hooks fire

        # Global average pooling of gradients  →  (C,)
        weights = self._gradients.mean(dim=(2, 3))  # (1, C)

        # Weighted combination of activation maps  →  (H, W)
        cam = (weights[0, :, None, None] * self._activations[0]).sum(dim=0)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Upsample to input resolution (32×32)
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)        # (1,1,h,w)
        cam_up = F.interpolate(cam_t, size=(32, 32), mode='bilinear',
                               align_corners=False)
        return cam_up.squeeze().numpy()

    def remove(self):
        """MUST be called after use to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _get_last_conv(model: FullModel) -> torch.nn.Module:
    """Return the last Conv2d in the backbone (layer4[-1].conv2 for ResNet18)."""
    last_conv = None
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d found in backbone")
    return last_conv


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Image selection  (24 test images across 3 groups)
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def select_images(model: FullModel, device: torch.device) \
        -> tuple[list[dict], np.ndarray]:
    """
    Returns
    -------
    selected : list of 24 dicts with keys
               image_tensor, image_rgb, soft_label, hard_label,
               global_idx, group, entropy
    soft_labels_all : (10000, 10) array for the whole CIFAR-10H test set
    """
    # ── load soft labels (all 10 000 CIFAR-10 test images) ──────────────────
    probs_path = os.path.join(cfg['data_dir'], cfg['cifar10h_probs_file'])
    soft_labels_all = np.load(probs_path).astype(np.float32)   # (10000, 10)
    entropy_all = compute_entropy(soft_labels_all)             # (10000,)

    # ── load raw CIFAR-10 test set (no transform, for RGB display) ───────────
    cifar_raw = datasets.CIFAR10(
        root=cfg['data_dir'], train=False, download=True, transform=None)

    # ── build normalised tensor for model inference ──────────────────────────
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg['normalization_mean'], cfg['normalization_std']),
    ])

    # ── run full-test-set inference to find disagreement cases ───────────────
    print("  Running inference on full test set for image selection…")
    model.eval()
    all_preds_cls = []
    all_logits    = []

    with torch.no_grad():
        for i in range(0, 10000, 256):
            batch_tensors = torch.stack([
                eval_tf(cifar_raw[j][0]) for j in range(i, min(i + 256, 10000))
            ]).to(device)
            logits = model(batch_tensors)
            all_preds_cls.append(logits.argmax(dim=1).cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    preds_cls = np.concatenate(all_preds_cls)   # (10000,)
    logits_np = np.concatenate(all_logits)      # (10000, 10)
    hard_labels = np.array([cifar_raw[i][1] for i in range(10000)])

    # ── group A : 8 highest entropy ──────────────────────────────────────────
    high_idx = np.argsort(-entropy_all)[:8].tolist()

    # ── group B : 8 lowest entropy (but entropy > 0 to avoid degenerate) ─────
    positive_entropy_mask = entropy_all > 0.001
    low_candidates = np.where(positive_entropy_mask)[0]
    low_idx = low_candidates[np.argsort(entropy_all[low_candidates])[:8]].tolist()

    # ── group C : disagreement cases ─────────────────────────────────────────
    # model predicted class ≠ hard label AND dominant soft label ≠ hard label
    disagree_mask = (preds_cls != hard_labels)
    # soft-label dominant class also disagrees with hard label
    soft_dominant = soft_labels_all.argmax(axis=1)
    interesting   = disagree_mask & (soft_dominant != hard_labels)
    # pick the 8 with highest soft-label probability for the predicted class
    # (these are the most justified disagreements)
    interesting_idxs = np.where(interesting)[0]
    pred_confidence  = logits_np[interesting_idxs,
                                  preds_cls[interesting_idxs]]
    top8_within = np.argsort(-pred_confidence)[:8]
    disagree_idx = interesting_idxs[top8_within].tolist()

    # ── assemble 24 records ───────────────────────────────────────────────────
    groups = (
        [(i, 'high_entropy')   for i in high_idx] +
        [(i, 'low_entropy')    for i in low_idx] +
        [(i, 'disagreement')   for i in disagree_idx]
    )

    selected = []
    for global_idx, group in groups:
        raw_img, hard_lbl = cifar_raw[global_idx]
        img_rgb    = np.array(raw_img)                    # (32,32,3) uint8
        img_tensor = eval_tf(raw_img).unsqueeze(0)        # (1,3,32,32)

        selected.append(dict(
            image_tensor = img_tensor,
            image_rgb    = img_rgb,
            soft_label   = soft_labels_all[global_idx],
            hard_label   = hard_lbl,
            global_idx   = global_idx,
            group        = group,
            entropy      = float(entropy_all[global_idx]),
            pred_class   = int(preds_cls[global_idx]),
            confidence   = float(logits_np[global_idx, preds_cls[global_idx]]),
        ))

    return selected, soft_labels_all


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Figure rendering
# ══════════════════════════════════════════════════════════════════════════════

def _render_figure(record: dict, heatmap: np.ndarray,
                   out_dir: str, loss_name: str, fig_idx: int) -> str:
    """
    3-panel figure:
      Panel 1 — original image
      Panel 2 — Grad-CAM heatmap (jet colormap)
      Panel 3 — overlay (heatmap alpha-blended on image)
    Bottom sub-panel: soft-label distribution bar chart
    """
    img_rgb   = record['image_rgb']           # (32,32,3) uint8
    soft      = record['soft_label']          # (10,) float
    hard_lbl  = record['hard_label']
    pred_cls  = record['pred_class']
    conf      = record['confidence']
    entropy   = record['entropy']
    group     = record['group']
    gidx      = record['global_idx']

    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], hspace=0.35, wspace=0.25)

    # ── panel 1: original ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title(f'Original\nTrue: {CLASSES[hard_lbl]}', fontsize=9, fontweight='bold')
    ax1.axis('off')

    # ── panel 2: heatmap ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    ax2.set_title(f'Grad-CAM ({loss_name})\nPred: {CLASSES[pred_cls]}  conf={conf:.2f}',
                  fontsize=9, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # ── panel 3: overlay ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_rgb)
    ax3.imshow(heatmap, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    ax3.set_title(f'Overlay\nEntropy={entropy:.3f}', fontsize=9, fontweight='bold')
    ax3.axis('off')

    # ── bottom panel: soft-label bar ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    colors = ['#e74c3c' if i == hard_lbl else
              '#2ecc71' if i == pred_cls else
              '#3498db' for i in range(10)]
    bars = ax4.bar(CLASSES, soft, color=colors, edgecolor='white', linewidth=0.5)
    ax4.set_ylim(0, max(soft.max() * 1.2, 0.15))
    ax4.set_ylabel('Soft-label prob', fontsize=8)
    ax4.set_title('Human soft-label distribution', fontsize=8)
    ax4.tick_params(axis='x', labelsize=7, rotation=30)
    ax4.tick_params(axis='y', labelsize=7)

    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='True hard label'),
        mpatches.Patch(color='#2ecc71', label='Model prediction'),
        mpatches.Patch(color='#3498db', label='Other class'),
    ]
    ax4.legend(handles=legend_patches, fontsize=7, loc='upper right')

    group_tag = group.replace('_', ' ').title()
    fig.suptitle(f'[{group_tag}]  idx={gidx}  {loss_name} Grad-CAM',
                 fontsize=10, fontweight='bold', y=1.01)

    fname = f'{fig_idx:02d}_{group}_{CLASSES[hard_lbl]}_idx{gidx}.png'
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return fpath


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_gradcam(kl_ckpt_path: str, jsd_ckpt_path: str,
                device: torch.device | None = None) -> list[dict]:
    """
    Runs the full Grad-CAM pipeline for both KL and JSD checkpoints.

    Returns
    -------
    records : the 24 selected image metadata dicts (with saved_paths added)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Grad-CAM] Using device: {device}")

    # ── load both models ─────────────────────────────────────────────────────
    print("\n[Grad-CAM] Loading KL model…")
    kl_model,  kl_cfg  = load_model(kl_ckpt_path,  device)
    print("[Grad-CAM] Loading JSD model…")
    jsd_model, jsd_cfg = load_model(jsd_ckpt_path, device)

    # ── select 24 images (using KL model for disagreement detection) ──────────
    print("\n[Grad-CAM] Selecting 24 images…")
    selected, _ = select_images(kl_model, device)
    print(f"  Groups: "
          f"{sum(1 for r in selected if r['group']=='high_entropy')} high-entropy, "
          f"{sum(1 for r in selected if r['group']=='low_entropy')} low-entropy, "
          f"{sum(1 for r in selected if r['group']=='disagreement')} disagreement")

    # ── attach last conv layer targets ───────────────────────────────────────
    kl_last_conv  = _get_last_conv(kl_model)
    jsd_last_conv = _get_last_conv(jsd_model)

    kl_gcam  = GradCAM(kl_model,  kl_last_conv)
    jsd_gcam = GradCAM(jsd_model, jsd_last_conv)

    saved_paths_kl  = []
    saved_paths_jsd = []

    print("\n[Grad-CAM] Generating figures…")
    for i, rec in enumerate(selected):
        x = rec['image_tensor'].to(device)

        # ── KL ───────────────────────────────────────────────────────────────
        kl_model.zero_grad()
        logits_kl    = kl_model(x)
        pred_kl      = int(logits_kl.argmax(dim=1).item())
        rec['pred_class']  = pred_kl
        rec['confidence']  = float(logits_kl[0, pred_kl].item())

        heatmap_kl = kl_gcam.compute(x, pred_kl)
        path_kl    = _render_figure(rec, heatmap_kl, OUT_KL,
                                    loss_name='KL', fig_idx=i + 1)
        saved_paths_kl.append(path_kl)

        # ── JSD ──────────────────────────────────────────────────────────────
        jsd_model.zero_grad()
        logits_jsd   = jsd_model(x)
        pred_jsd     = int(logits_jsd.argmax(dim=1).item())
        rec_jsd = dict(rec)          # shallow copy — don't mutate original
        rec_jsd['pred_class']  = pred_jsd
        rec_jsd['confidence']  = float(logits_jsd[0, pred_jsd].item())

        heatmap_jsd = jsd_gcam.compute(x, pred_jsd)
        path_jsd    = _render_figure(rec_jsd, heatmap_jsd, OUT_JSD,
                                     loss_name='JSD', fig_idx=i + 1)
        saved_paths_jsd.append(path_jsd)

        group_short = rec['group'][:4].upper()
        print(f"  [{i+1:02d}/24] {group_short}  idx={rec['global_idx']:5d}  "
              f"H={rec['entropy']:.3f}  saved ✓")

    # ── CRITICAL: remove all hooks ────────────────────────────────────────────
    kl_gcam.remove()
    jsd_gcam.remove()
    print("\n[Grad-CAM] All hooks removed.")

    for i, rec in enumerate(selected):
        rec['kl_fig_path']  = saved_paths_kl[i]
        rec['jsd_fig_path'] = saved_paths_jsd[i]

    print(f"\n[Grad-CAM] Done. Figures saved to:")
    print(f"  KL  → {OUT_KL}")
    print(f"  JSD → {OUT_JSD}")
    return selected


if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(__file__), '..', 'module3', 'outputs')
    kl_path  = os.path.join(BASE, 'checkpoint_KL_A_pretrained.pt')
    jsd_path = os.path.join(BASE, 'checkpoint_JSD_A_pretrained.pt')
    run_gradcam(kl_path, jsd_path)