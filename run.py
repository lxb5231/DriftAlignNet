"""Train or evaluate DriftAlignNet for multivariate time-series forecasting."""

import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false or 1/0.")


def build_parser():
    parser = argparse.ArgumentParser(description="DriftAlignNet forecasting")
    parser.add_argument("--is_training", type=int, choices=[0, 1], required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model", choices=["DriftAlignNet"], default="DriftAlignNet")
    parser.add_argument("--random_seed", type=int, default=2024)
    parser.add_argument(
        "--data",
        required=True,
        choices=[
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "custom",
            "RealE",
            "energy",
            "power",
        ],
    )
    parser.add_argument("--root_path", default="./dataset/")
    parser.add_argument("--data_path", default="ETTh2.csv")
    parser.add_argument("--features", choices=["M", "S", "MS"], default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument(
        "--embed", choices=["timeF", "fixed", "learned"], default="timeF"
    )
    parser.add_argument("--checkpoints", default="./checkpoints/")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--cycle", type=int, default=24)
    parser.add_argument("--anchor_cycle", type=int)
    parser.add_argument("--local_cycle", type=int)
    parser.add_argument("--use_revin", type=int, choices=[0, 1], default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--raw_weight", type=float, default=0.5)
    parser.add_argument("--diff_weight", type=float, default=0.5)
    parser.add_argument("--similarity_eps", type=float, default=1e-5)
    parser.add_argument(
        "--selector_type",
        default="fast_exact_directional_lag_mmr",
        choices=[
            "statistical",
            "directional_lag_mmr",
            "efficient_directional_lag_mmr",
            "fast_exact_directional_lag_mmr",
        ],
    )
    parser.add_argument("--max_lag", type=int, default=8)
    parser.add_argument("--candidate_multiplier", type=int, default=8)
    parser.add_argument("--candidate_budget", type=int, default=16)
    parser.add_argument("--lag_candidate_budget", type=int, default=4)
    parser.add_argument("--selector_target_chunk_size", type=int, default=16)
    parser.add_argument("--selector_candidate_chunk_size", type=int, default=32)
    parser.add_argument("--fast_strict_legacy_threshold", type=int, default=16)
    parser.add_argument("--fast_global_candidate_budget", type=int, default=64)
    parser.add_argument("--fast_refine_budget", type=int, default=32)
    parser.add_argument("--global_graph_sample_stride", type=int, default=24)
    parser.add_argument("--global_graph_max_windows", type=int, default=1024)
    parser.add_argument("--global_graph_cache", type=str_to_bool, default=True)
    parser.add_argument("--lag_penalty", type=float, default=0.1)
    parser.add_argument("--stability_windows", type=int, default=4)
    parser.add_argument("--stability_gamma", type=float, default=5.0)
    parser.add_argument("--mmr_lambda", type=float, default=0.75)
    parser.add_argument("--redundancy_threshold", type=float)
    parser.add_argument(
        "--alignment_padding", choices=["replicate", "zero"], default="replicate"
    )
    parser.add_argument(
        "--predict_type", choices=["CycleAligned"], default="CycleAligned"
    )
    parser.add_argument("--cycle_forecaster_hidden", type=int, default=128)
    parser.add_argument("--transient_recent_cycles", type=int, default=2)
    parser.add_argument("--transient_gate_bias", type=float, default=-2.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--lradj",
        choices=[
            "type1",
            "type2",
            "type3",
            "cosine",
            "constant",
            "3",
            "4",
            "5",
            "6",
            "TST",
        ],
        default="type3",
    )
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_gpu", type=str_to_bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", default="0")
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Save predictions for the test split after evaluation.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    import random
    import numpy as np
    import torch
    from exp.exp_main import Exp_Main

    if (
        min(
            args.seq_len,
            args.pred_len,
            args.enc_in,
            args.batch_size,
            args.train_epochs,
            args.itr,
            args.patience,
        )
        < 1
    ):
        raise ValueError(
            "Lengths, channels, batch size, epochs, repetitions and patience must be positive."
        )
    if args.label_len < 0 or args.label_len > args.seq_len:
        raise ValueError("label_len must lie between zero and seq_len.")
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    if args.use_amp and not args.use_gpu:
        raise ValueError("--use_amp requires CUDA; disable it for CPU execution.")
    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = [int(device.strip()) for device in args.devices.split(",")]
        args.gpu = args.device_ids[0]
    print(args)
    for iteration in range(args.itr):
        seed = args.random_seed + iteration
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        setting = (
            f"{args.model_id}_{args.model}_{args.data}_ft{args.features}"
            f"_sl{args.seq_len}_pl{args.pred_len}_cycle{args.cycle}_seed{seed}"
        )
        experiment = Exp_Main(args)
        if args.is_training:
            experiment.train(setting)
        experiment.test(setting, test=not args.is_training)
        if args.do_predict:
            experiment.predict(setting, load=True)
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
