# DriftAlignNet

Code for multivariate time-series forecasting with DriftAlignNet. This repository includes the model, training and evaluation entry points, data loaders, and main experiment configurations.

## Installation

Python 3.11 is recommended. Python 3.10 is also supported.

```bash
conda create -n driftalignnet python=3.11 -y
conda activate driftalignnet
python -m pip install --upgrade pip
```

For CPU execution:

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.1:

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Alternatively, run `conda env create -f environment.yml` to install the default PyTorch environment. GPU execution requires a compatible NVIDIA driver.

## Data Preparation

Download the datasets separately; they are not included in this repository.

- ETT: [ETDataset](https://github.com/zhouhaoyi/ETDataset).
- Other forecasting datasets: [Download datasets](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view).

Place the data files directly in `dataset/`, for example:

```text
dataset/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── electricity.csv
├── exchange_rate.csv
├── weather.csv
└── illness.csv
```

To download ETTh2 using Bash:

```bash
mkdir -p dataset
curl -fL https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv -o dataset/ETTh2.csv
```

Sort input data chronologically. ETT and standard custom CSV files should contain a `date` column and numeric channels. The ETT loaders use the first 12 months for training, the next 4 months for validation, and the following 4 months for testing, with each month defined as 30 days. Standard `custom` datasets use an approximately 70% / 10% / 20% chronological split. The scaler is fitted only on the training split.

See `scripts/DriftAlignNet/configs.json` for dataset filenames, channel counts, and loader types. Data loading code is located in `data_provider/`.

## Training and Evaluation

Run all commands from the repository root. To train on ETTh2 with an input length of 96 and a prediction length of 96:

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --pred_len 96 --seed 2024
```

Training selects the best checkpoint using validation loss and evaluates it on the test split after training. An available CUDA device is used by default. To force CPU execution:

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --pred_len 96 --seed 2024 --use_gpu false
```

To evaluate the best checkpoint saved with the same configuration:

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --pred_len 96 --seed 2024 --is_training 0
```

Omit `--pred_len` to run all configured prediction lengths, or omit `--seed` to run all configured seeds for the dataset. To preview the commands without running experiments:

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --dry_run
```

The Bash and PowerShell wrappers share the same configuration:

```bash
bash scripts/DriftAlignNet/etth2.sh --pred_len 96 --seed 2024
bash run_main.sh etth2 --pred_len 96 --seed 2024
```

```powershell
.\scripts\DriftAlignNet\etth2.ps1 --pred_len 96 --seed 2024
.\run_main.ps1 etth2 --pred_len 96 --seed 2024
```

Available configurations are `etth1`, `etth2`, `ettm1`, `ettm2`, `electricity`, `exchange_rate`, `weather`, `solar`, `traffic`, `ili`, `france`, `germany`, `energy`, and `power`. Both shells use a single configuration source based on the existing main experiment Bash settings, except ETTm2, which uses the existing full CycleAligned PowerShell settings. For ETTh1, prediction lengths 192 and 336 use the full model with `top_k=2`, and `use_revin` is consistently set to 1.

For custom training settings, invoke the entry point directly:

```bash
python run.py --is_training 1 --model_id ETTh2_96_96 --model DriftAlignNet --data ETTh2 --root_path ./dataset/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 96 --enc_in 7 --cycle 24 --anchor_cycle 24 --local_cycle 24 --top_k 1 --d_model 128 --dropout 0.3 --use_revin 1 --similarity_eps 1e-4 --selector_type fast_exact_directional_lag_mmr --batch_size 256 --learning_rate 0.001 --train_epochs 50 --patience 10 --random_seed 2024
```

Run `python run.py --help` for all options. Reduce `--batch_size` if GPU memory is insufficient. Standalone evaluation requires the same model parameters, data settings, `model_id`, and seed used during training. The `--do_predict` option saves an additional prediction file for the test split; the current loader uses the test split rather than unseen future data.

## Local Outputs

| Output | Path |
| --- | --- |
| Best checkpoint | `checkpoints/<setting>/checkpoint.pth` |
| Test predictions and ground truth | `results/<setting>/pred.npy`, `true.npy` |
| Optional additional predictions | `results/<setting>/real_prediction.npy` |
| Main experiment logs | `logs/DriftAlignNet/<dataset>/` |
| MSE and MAE summary | `result.txt` |
| Training channel graph cache | `cache/channel_graph/` |

MSE and MAE are computed on the standardized scale by default. Outputs, datasets, checkpoints, and caches are excluded by `.gitignore`.

## Repository Structure

```text
models/DriftAlignNet.py       Main model
module/channel_selectors.py  Channel selection and training candidate graphs
module/predict_head/         Cycle-aligned prediction head
module/cuda_graph_runner.py  CUDA Graph inference
exp/                        Training, validation, and evaluation
data_provider/predict/       Data loading
utils/                      Shared utilities
scripts/DriftAlignNet/       Main experiment configurations and launchers
run.py                      Forecasting entry point
```

See [LICENSE](LICENSE) for licensing information.
