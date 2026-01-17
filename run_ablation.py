import os
import sys
import json
import argparse
import subprocess
from datetime import datetime


def run(cmd, cwd=None):
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for supervisor additions (flow + temporal + direction).")
    parser.add_argument("--base-cmd", type=str, default=None,
                        help="Optional: full base command string (excluding python main.py). If not set, use flags below.")
    parser.add_argument("--dataset", default="wrongway-dataset")
    parser.add_argument("--rgb-list", default="list/mytrain.list")
    parser.add_argument("--test-rgb-list", default="list/mytest.list")
    parser.add_argument("--feature-size", type=int, default=1024)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--lr", type=str, default="[0.0001]*15000")
    parser.add_argument("--split-by-label", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")

    # Supervisor additions default strengths (can override)
    parser.add_argument("--flow-suffix", type=str, default="_flow")
    parser.add_argument("--flow-dim", type=int, default=4)
    parser.add_argument("--lambda-temp", type=float, default=1e-4)
    parser.add_argument("--lambda-dir", type=float, default=1e-2)
    parser.add_argument("--dir-margin", type=float, default=0.0)
    parser.add_argument("--temp-on-all", action="store_true")
    parser.add_argument("--temp-norm-dim", action="store_true",
                        help="Also normalize temporal smoothness by D (recommended).")

    # Visdom
    parser.add_argument("--viz-server", type=str, default="http://localhost")
    parser.add_argument("--viz-port", type=int, default=8097)
    parser.add_argument("--viz-env", type=str, default="wrongway-ablation",
                        help="Base Visdom environment prefix. Each run will append a suffix.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visdom (still logs to output dirs).")

    parser.add_argument("--runs-dir", type=str, default="runs_ablation",
                        help="Directory to write per-run outputs (ckpt/logs).")

    args = parser.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)

    # Define ablation matrix
    # A0: RGB only (no flow, no temp, no dir)
    # A1: +Flow (feature concat only; no temp, no dir)
    # A2: +Flow +Temp
    # A3: +Flow +Dir
    # A4: +Flow +Temp +Dir (full)
    matrix = [
        ("A0_rgb_only", dict(use_flow=False, lambda_temp=0.0, lambda_dir=0.0)),
        ("A1_flow_only", dict(use_flow=True, lambda_temp=0.0, lambda_dir=0.0)),
        ("A2_flow_temp", dict(use_flow=True, lambda_temp=args.lambda_temp, lambda_dir=0.0)),
        ("A3_flow_dir", dict(use_flow=True, lambda_temp=0.0, lambda_dir=args.lambda_dir)),
        ("A4_flow_temp_dir", dict(use_flow=True, lambda_temp=args.lambda_temp, lambda_dir=args.lambda_dir)),
    ]

    # Build base args list
    base = [
        "--dataset", args.dataset,
        "--rgb-list", args.rgb_list,
        "--test-rgb-list", args.test_rgb_list,
        "--feature-size", str(args.feature_size),
        "--num-segments", str(args.num_segments),
        "--batch-size", str(args.batch_size),
        "--max-epoch", str(args.max_epoch),
        "--lr", args.lr,
        "--seed", str(args.seed),
    ]
    if args.deterministic:
        base += ["--deterministic"]
    if args.split_by_label:
        base += ["--split-by-label"]
    if args.temp_on_all:
        base += ["--temp-on-all"]
    if args.temp_norm_dim:
        base += ["--temp-norm-dim"]
    if args.no_viz:
        base += ["--no-viz"]
    else:
        base += ["--viz-server", args.viz_server, "--viz-port", str(args.viz_port)]

    summary = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for tag, cfg in matrix:
        out_dir = os.path.join(args.runs_dir, f"{stamp}_{tag}")
        os.makedirs(out_dir, exist_ok=True)

        run_args = base + ["--output-dir", out_dir, "--model-name", tag]

        if cfg["use_flow"]:
            run_args += ["--use-flow", "--flow-suffix", args.flow_suffix, "--flow-dim", str(args.flow_dim)]
        run_args += ["--lambda-temp", str(cfg["lambda_temp"])]
        run_args += ["--lambda-dir", str(cfg["lambda_dir"]), "--dir-margin", str(args.dir_margin)]

        # Per-run Visdom env
        if not args.no_viz:
            run_args += ["--viz-env", f"{args.viz_env}-{tag}"]

        cmd = [sys.executable, "main.py"] + run_args
        run(cmd)

        # Capture best AUC file if present
        best_path = os.path.join(out_dir, "AUC-per-epoch", "best_auc.txt")
        best = {}
        if os.path.exists(best_path):
            with open(best_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        best[k] = v
        summary.append({"run": tag, "out_dir": out_dir, **best})

        with open(os.path.join(args.runs_dir, f"{stamp}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\n=== Ablation Summary ===")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
