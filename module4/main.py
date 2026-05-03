"""
Module 4 — Entry Point
Usage:
    python module4/main.py
    python module4/main.py --kl_checkpoint path/to/kl.pt
    python module4/main.py --jsd_checkpoint path/to/jsd.pt
    python module4/main.py --skip_gradcam     # report only (figures must exist)
    python module4/main.py --report_only      # alias for --skip_gradcam
"""

import os
import sys
import argparse
import time
import torch

# ── make sibling modules importable ──────────────────────────────────────────
MODULE4_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(MODULE4_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Default checkpoint locations (relative to repo root)
DEFAULT_KL  = os.path.join(ROOT_DIR, 'module3', 'outputs',
                            'checkpoint_KL_A_pretrained.pt')
DEFAULT_JSD = os.path.join(ROOT_DIR, 'module3', 'outputs',
                            'checkpoint_JSD_A_pretrained.pt')


def _banner(msg: str) -> None:
    width = 70
    print('\n' + '═' * width)
    print(f'  {msg}')
    print('═' * width)


def _check_file(path: str, label: str) -> bool:
    if not os.path.isfile(path):
        print(f"  [WARNING] {label} not found: {path}")
        return False
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [OK] {label}: {path}  ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Module 4 — Grad-CAM & Report Generator')
    parser.add_argument('--kl_checkpoint',  default=DEFAULT_KL,
                        help='Path to KL fine-tuned checkpoint (.pt)')
    parser.add_argument('--jsd_checkpoint', default=DEFAULT_JSD,
                        help='Path to JSD fine-tuned checkpoint (.pt)')
    parser.add_argument('--skip_gradcam',   action='store_true',
                        help='Skip Grad-CAM step; generate report from existing figures')
    parser.add_argument('--report_only',    action='store_true',
                        help='Alias for --skip_gradcam')
    parser.add_argument('--device',         default=None,
                        help='Force device: "cpu" or "cuda". Auto-detected if omitted.')
    args = parser.parse_args()

    skip_gradcam = args.skip_gradcam or args.report_only

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _banner('Module 4 — Starting')
    print(f'  Device  : {device}')
    print(f'  KL ckpt : {args.kl_checkpoint}')
    print(f'  JSD ckpt: {args.jsd_checkpoint}')
    print(f'  Mode    : {"report-only (no Grad-CAM)" if skip_gradcam else "full pipeline"}')

    records = []

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Grad-CAM
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_gradcam:
        _banner('Step 1/2 — Grad-CAM')

        # Pre-flight checks
        kl_ok  = _check_file(args.kl_checkpoint,  'KL checkpoint')
        jsd_ok = _check_file(args.jsd_checkpoint, 'JSD checkpoint')

        if not (kl_ok and jsd_ok):
            print('\n  [ERROR] One or more checkpoint files are missing.')
            print('  Run with --report_only to skip Grad-CAM and use existing figures.')
            sys.exit(1)

        t0 = time.time()

        # Import here (not at top-level) so that --report_only never triggers
        # model loading or CUDA initialisation.
        from gradcam import run_gradcam

        try:
            records = run_gradcam(
                kl_ckpt_path  = args.kl_checkpoint,
                jsd_ckpt_path = args.jsd_checkpoint,
                device        = device,
            )
        except Exception as exc:
            print(f'\n  [ERROR] Grad-CAM failed: {exc}')
            import traceback; traceback.print_exc()
            print('\n  Continuing to report generation with any figures that were saved.')

        elapsed = time.time() - t0
        print(f'\n  Grad-CAM completed in {elapsed:.1f}s  '
              f'({len(records)} image records returned)')

    else:
        _banner('Step 1/2 — Grad-CAM [SKIPPED]')
        print('  Using figures already present in module4/figures/')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Report
    # ══════════════════════════════════════════════════════════════════════════
    _banner('Step 2/2 — Report Generation')

    from report_generator import generate_report

    t0 = time.time()
    try:
        report_path = generate_report(records=records if records else None)
    except Exception as exc:
        print(f'\n  [ERROR] Report generation failed: {exc}')
        import traceback; traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f'  Report completed in {elapsed:.1f}s')

    # ══════════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════════
    _banner('Module 4 — Complete')
    print(f'  Report : {report_path}')
    kl_fig_dir  = os.path.join(MODULE4_DIR, 'figures', 'gradcam')
    jsd_fig_dir = os.path.join(MODULE4_DIR, 'figures', 'gradcam_JSD')
    n_kl  = len([f for f in os.listdir(kl_fig_dir)  if f.endswith('.png')]) \
            if os.path.isdir(kl_fig_dir)  else 0
    n_jsd = len([f for f in os.listdir(jsd_fig_dir) if f.endswith('.png')]) \
            if os.path.isdir(jsd_fig_dir) else 0
    print(f'  KL  figures : {n_kl}  → {kl_fig_dir}')
    print(f'  JSD figures : {n_jsd} → {jsd_fig_dir}')
    print()


if __name__ == '__main__':
    main()