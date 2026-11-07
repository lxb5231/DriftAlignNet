# DriftAlignNet

DriftAlignNet 的多变量时间序列预测代码。发布内容包含主模型、训练和测试入口、数据加载器，以及主实验配置。

## 环境安装

推荐 Python 3.11，也可使用 Python 3.10。

```bash
conda create -n driftalignnet python=3.11 -y
conda activate driftalignnet
python -m pip install --upgrade pip
```

CPU 环境：

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CUDA 12.1 环境：

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

也可执行 `conda env create -f environment.yml` 安装默认 PyTorch 环境。GPU 运行需要兼容的 NVIDIA 驱动。

## 数据准备

数据集需要自行下载，发布仓库不包含数据。

- ETT：[ETDataset](https://github.com/zhouhaoyi/ETDataset)。
- 其他预测数据集：[数据下载地址](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view)。

将数据文件直接放入 `dataset/`，例如：

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

例如下载 ETTh2（Bash）：

```bash
mkdir -p dataset
curl -fL https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv -o dataset/ETTh2.csv
```

输入数据须按时间排序，ETT 和普通自定义 CSV 保留 `date` 列及数值通道。ETT 加载器使用前 12 个月训练、接下来各 4 个月验证和测试，每月按 30 天计算。普通 `custom` 数据按时间以约 70% / 10% / 20% 划分；标准化器仅在训练集拟合。

其他数据集的文件名、通道数和加载器类型见 `scripts/DriftAlignNet/configs.json`。`data_provider/` 保存数据加载代码。

## 训练与测试

以下命令均在本仓库根目录执行。ETTh2、输入长度 96、预测长度 96 的正式配置：

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --pred_len 96 --seed 2024
```

训练以验证集损失选择最佳权重，并在训练结束后执行测试。默认使用可用的 CUDA 设备；强制使用 CPU：

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --pred_len 96 --seed 2024 --use_gpu false
```

加载同一配置保存的最佳权重单独测试：

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --pred_len 96 --seed 2024 --is_training 0
```

省略 `--pred_len` 或 `--seed`，分别运行该数据集配置中的全部预测长度或全部种子。先检查将要执行的命令：

```bash
python scripts/DriftAlignNet/run_dataset.py etth2 --dry_run
```

Bash 和 PowerShell 包装脚本共用同一份配置：

```bash
bash scripts/DriftAlignNet/etth2.sh --pred_len 96 --seed 2024
bash run_main.sh etth2 --pred_len 96 --seed 2024
```

```powershell
.\scripts\DriftAlignNet\etth2.ps1 --pred_len 96 --seed 2024
.\run_main.ps1 etth2 --pred_len 96 --seed 2024
```

配置项包括 `etth1`、`etth2`、`ettm1`、`ettm2`、`electricity`、`exchange_rate`、`weather`、`solar`、`traffic`、`ili`、`france`、`germany`、`energy` 和 `power`。两种 Shell 共用一份配置：默认沿用现有主实验 Bash 参数；ETTm2 采用现有 PowerShell 中的完整 CycleAligned 配置。ETTh1 的 192、336 预测长度使用完整模型 `top_k=2`，其 `use_revin` 统一为 1。

自定义训练可直接调用入口：

```bash
python run.py --is_training 1 --model_id ETTh2_96_96 --model DriftAlignNet --data ETTh2 --root_path ./dataset/ --data_path ETTh2.csv --features M --seq_len 96 --pred_len 96 --enc_in 7 --cycle 24 --anchor_cycle 24 --local_cycle 24 --top_k 1 --d_model 128 --dropout 0.3 --use_revin 1 --similarity_eps 1e-4 --selector_type fast_exact_directional_lag_mmr --batch_size 256 --learning_rate 0.001 --train_epochs 50 --patience 10 --random_seed 2024
```

全部参数见 `python run.py --help`。显存不足时降低 `--batch_size`。单独测试须保持模型参数、数据配置、`model_id` 和种子与训练一致。`--do_predict` 额外保存测试集预测；当前加载器使用测试切分，不读取未知未来数据。

## 本地输出

| 内容 | 路径 |
| --- | --- |
| 最佳权重 | `checkpoints/<实验标识>/checkpoint.pth` |
| 测试预测与真实值 | `results/<实验标识>/pred.npy`、`true.npy` |
| 可选额外预测 | `results/<实验标识>/real_prediction.npy` |
| 主实验日志 | `logs/DriftAlignNet/<数据集>/` |
| MSE、MAE 汇总 | `result.txt` |
| 训练集通道图缓存 | `cache/channel_graph/` |

MSE、MAE 默认在标准化后的尺度上计算。输出、数据、权重和缓存均已由 `.gitignore` 排除。

## 代码结构

```text
models/DriftAlignNet.py       主模型
module/channel_selectors.py  通道选择与训练集候选图
module/predict_head/         周期对齐预测头
module/cuda_graph_runner.py  CUDA Graph 推理
exp/                        训练、验证与测试
data_provider/predict/       数据加载
utils/                      必要公共工具
scripts/DriftAlignNet/       主实验配置与启动脚本
run.py                      预测任务入口
```

许可证见 [LICENSE](LICENSE)。
