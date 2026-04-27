import os
import sys
import argparse
import subprocess

from config import config


def run_cmd(cmd):
    """Helper to run shell commands and check for errors."""
    print(f"\n>> Running: {' '.join(cmd)}")
    print("\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nError: Command failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses', nargs='+', default=None, help='Subset of losses to run')
    parser.add_argument('--heads', nargs='+', default=None, help='Subset of heads to run')
    parser.add_argument('--skip-finetune', action='store_true', help='Skip training')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--skip-viz', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    # Defaults from config if not specified
    losses = args.losses if args.losses else config['loss_functions']
    heads = args.heads if args.heads else [config['primary_head']]
    
    print("\n--- Pipeline Started ---")
    print(f"Target Losses: {losses}")
    print(f"Target Heads:  {heads}")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Step 1: Training
    if not args.skip_finetune:
        print("\n[Step 1] Fine-tuning model variants...")
        for loss in losses:
            for head in heads:
                # Main run with pretrained backbone
                run_cmd([
                    'python', 'finetune.py',
                    '--loss', loss,
                    '--head', head,
                    '--init', 'pretrained'
                ])
                
                # Ablation: Random init for KL + Head A
                if head == config['primary_head'] and loss == 'KL':
                    run_cmd([
                        'python', 'finetune.py',
                        '--loss', loss,
                        '--head', head,
                        '--init', 'random'
                    ])
    
    # Step 2: Evaluation
    if not args.skip_eval:
        print("\n[Step 2] Running evaluation on all checkpoints...")
        run_cmd(['python', 'evaluate.py', '--all'])
    
    # Step 3: Plots
    if not args.skip_viz:
        print("\n[Step 3] Generating visualizations...")
        run_cmd(['python', 'visualize.py', '--all'])
    
    print("\nPipeline finished successfully.")
    print(f"Results are in: {config['output_dir']}")


if __name__ == '__main__':
    main()