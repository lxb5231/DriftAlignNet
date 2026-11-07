"""Shared main-experiment launcher for Bash, PowerShell and Python."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = json.loads(
    Path(__file__).with_name("configs.json").read_text(encoding="utf-8")
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the published DriftAlignNet configurations."
    )
    parser.add_argument("dataset", choices=sorted(CONFIGS))
    parser.add_argument(
        "--pred_len",
        type=int,
        help="Run one horizon; default: all configured horizons.",
    )
    parser.add_argument(
        "--seed", type=int, help="Run one seed; default: all configured seeds."
    )
    parser.add_argument("--root_path", default="./dataset/")
    parser.add_argument("--is_training", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_gpu", choices=["true", "false"], default="true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without training or creating logs.",
    )
    args = parser.parse_args()
    config = CONFIGS[args.dataset]
    rows = [
        row
        for row in config["horizons"]
        if args.pred_len is None or int(row["pred_len"]) == args.pred_len
    ]
    if not rows:
        parser.error(
            "The selected prediction horizon is not configured for this dataset."
        )
    seeds = config["seeds"] if args.seed is None else [args.seed]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for row in rows:
        for seed in seeds:
            params = {
                **config["common"],
                **row,
                "model": "DriftAlignNet",
                "root_path": args.root_path,
                "is_training": args.is_training,
                "random_seed": seed,
                "use_gpu": args.use_gpu,
                "gpu": args.gpu,
            }
            params["model_id"] = (
                f"{args.dataset}_{params['seq_len']}_{params['pred_len']}"
                f"_k{params['top_k']}_gc{params['anchor_cycle']}"
                f"_lc{params['local_cycle']}_selector{params['selector_type']}_seed{seed}"
            )
            command = [sys.executable, "-u", str(ROOT / "run.py")]
            for key, value in params.items():
                command.extend([f"--{key}", str(value)])
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            log_dir = ROOT / "logs" / "DriftAlignNet" / args.dataset
            log_dir.mkdir(parents=True, exist_ok=True)
            mode = "train" if args.is_training else "test"
            with (log_dir / f"{params['model_id']}_{mode}.log").open(
                "w", encoding="utf-8"
            ) as log:
                with subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                ) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = process.wait()
            if code:
                raise SystemExit(code)


if __name__ == "__main__":
    main()
