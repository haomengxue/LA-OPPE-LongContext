# LA-OPPE LongContext

面向长上下文建模的轻量位置增强方法：**Distance-Gated Residual LA-OPPE Score Bias**。

本项目在保留 Llama/RoPE 主路径的基础上，只在注意力分数上加入一个距离门控的残差位置偏置，用极少量新增参数增强长距离建模能力。仓库包含核心实现、训练脚本、评测脚本、已整理的实验结果，以及可选的轻量 delta 权重。

## 方法简介

RoPE 在常规上下文长度下表现稳定，但在更长上下文外推时可能退化。LA-OPPE Score Bias 的设计目标是：

- 不替换原始 RoPE；
- 不全量微调基础模型；
- 只在指定 Transformer 层加入轻量可学习 bias；
- 短距离尽量保持原模型行为，长距离才增强；
- 便于和 RoPE、PI、NTK、YaRN、ALiBi 等基线比较。

默认实验配置为 `LAOPPE_L2_th2048`：

- 注入层：最后两层，默认 `START_LAYER=14`、`END_LAYER=16`；
- 距离阈值：`THRESHOLD=2048`；
- 训练长度：`BLOCK_SIZE=4096`；
- 只训练新增 LA-OPPE 参数；
- 基础模型权重和第三方数据集不随仓库发布。

## 仓库结构

```text
LA-OPPE-LongContext/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   ├── laoppe_l2_th2048.yaml
│   ├── alibi_l2.yaml
│   ├── baselines.yaml
│   └── eval_config.yaml
├── src/
│   ├── llama_rope_laoppe_score_bias.py
│   ├── llama_rope_laoppe.py
│   └── utils.py
├── scripts/
│   ├── train_laoppe_score_bias.py
│   ├── train_alibi.py
│   ├── eval_ppl.py
│   ├── eval_needle.py
│   ├── eval_longbench_subset.py
│   ├── bclass_extra_experiments.py
│   ├── bclass_plus_experiments.py
│   └── make_tables_and_figures.py
├── results/
│   ├── ppl_results.csv
│   ├── ppl_sample_level.csv
│   ├── ppl_paired_ttest.csv
│   ├── needle_results.csv
│   ├── ppl8192_results.csv
│   ├── needle_yarn_results.csv
│   ├── longbench_subset_results.csv
│   └── final_experiment_summary.md
├── figures/
│   ├── attention_entropy_per_layer.png
│   ├── latency_by_length.png
│   └── throughput_by_length.png
├── docs/
│   └── experiment_notes.md
└── checkpoints/
    ├── README.md
    └── laoppe_l2_th2048_delta.pt
```

说明：

- `src/` 是方法核心实现；
- `scripts/` 里同时保留原始综合实验脚本和更易用的入口脚本；
- `results/` 是已经整理好的 CSV/Markdown 结果；
- `figures/` 是现有图表；
- `checkpoints/laoppe_l2_th2048_delta.pt` 是轻量 delta 权重，不包含基础模型；
- `docs/` 当前只放实验说明，论文 PDF/DOCX 如果后续公开，会放到这里。

## 不包含的内容


- `*.bak`、`*.bak_*`；
- AutoDL 容器路径截图；
- 未清洗的 `long_context_generations.jsonl`；
- Llama 或其他基础模型权重；
- Wikitext、LongBench 等第三方数据集原文件；
- 大体积训练中间 checkpoint。

这些文件已在 `.gitignore` 中尽量覆盖。

## 环境安装

建议使用 Linux 或 CUDA 可用的服务器环境。Windows 也可以浏览结果和阅读代码，但完整训练/长上下文评测建议放在 GPU 服务器上执行。

```bash
git clone https://github.com/<your-name>/LA-OPPE-LongContext.git
cd LA-OPPE-LongContext

conda create -n laoppe python=3.10 -y
conda activate laoppe

pip install -r requirements.txt
```

如果你的 CUDA/PyTorch 版本和 `requirements.txt` 不匹配，请先根据自己的显卡和 CUDA 版本安装对应 PyTorch，再安装剩余依赖。

## 准备基础模型

本仓库不提供 Llama 权重。请自行下载符合许可证要求的基础模型，例如 Llama 3.2 1B Base，然后设置环境变量：

```bash
export BASE_MODEL_PATH=/path/to/llama3.2-1b-base
export BASE_MODEL_DIR=$BASE_MODEL_PATH
```

如果你把模型放在仓库根目录下的 `baseline_model/`，也可以不设置 `BASE_MODEL_DIR`，脚本会优先尝试使用该目录。

Windows PowerShell 写法：

```powershell
$env:BASE_MODEL_PATH="D:\models\llama3.2-1b-base"
$env:BASE_MODEL_DIR=$env:BASE_MODEL_PATH
```

## 准备数据集

### 1. Wikitext

PPL 和训练脚本默认使用 Wikitext token 文件。推荐放置方式：

```text
data/
└── wikitext-103/
    ├── wiki.train.tokens
    └── wiki.valid.tokens
```

然后设置：

```bash
export TRAIN_DATA_PATH=./data/wikitext-103/wiki.train.tokens
export VALID_DATA_PATH=./data/wikitext-103/wiki.valid.tokens
export DATA_TRAIN=./data/wikitext-103/wiki.train.tokens
export DATA_VALID=./data/wikitext-103/wiki.valid.tokens
```

### 2. LongBench

LongBench 小子集评测默认读取本地 jsonl。推荐放置：

```text
LongBench/
├── passage_retrieval_en.jsonl
└── qasper.jsonl
```

然后设置：

```bash
export LONGBENCH_DIR=./LongBench
```

脚本也会尝试查找：

- `./LongBench/{task}.jsonl`
- `./LongBench/data/{task}.jsonl`
- `./LongBench/{task}/test.jsonl`

## 快速查看已有结果

如果你只想看本文已有实验结果，不需要下载模型和数据，直接查看：

```text
results/final_experiment_summary.md
results/ppl_results.csv
results/ppl_paired_ttest.csv
results/needle_results.csv
results/ppl8192_results.csv
results/needle_yarn_results.csv
results/longbench_subset_results.csv
```

图表在：

```text
figures/
```

## 使用已发布 delta 权重复现 LA-OPPE

仓库附带轻量 delta：

```text
checkpoints/laoppe_l2_th2048_delta.pt
```

设置：

```bash
export LAOPPE_STATE_DICT=./checkpoints/laoppe_l2_th2048_delta.pt
export START_LAYER=14
export END_LAYER=16
export THRESHOLD=2048
export TEMPERATURE=512
export ALPHA_MAX=0.05
```

然后运行评测。

## 训练 LA-OPPE

最小命令：

```bash
python scripts/train_laoppe_score_bias.py
```

推荐显式设置：

```bash
export BASE_MODEL_PATH=/path/to/llama3.2-1b-base
export TRAIN_DATA_PATH=./data/wikitext-103/wiki.train.tokens
export VALID_DATA_PATH=./data/wikitext-103/wiki.valid.tokens
export SAVE_DIR=./outputs/laoppe_l2_th2048

export BLOCK_SIZE=4096
export THRESHOLD=2048
export START_LAYER=14
export END_LAYER=16
export MAX_STEPS=1000
export LEARNING_RATE=1e-4
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8

python scripts/train_laoppe_score_bias.py
```

训练完成后，如果脚本保存了完整 state dict，可以将需要公开的轻量 delta 权重整理到 `checkpoints/`。不要上传完整基础模型权重。

## 训练 ALiBi 基线

```bash
export BASE_MODEL_PATH=/path/to/llama3.2-1b-base
export DATA_TRAIN=./data/wikitext-103/wiki.train.tokens
export ALIBI_SAVE_DIR=./outputs/alibi_l2

python scripts/train_alibi.py
```

等价于：

```bash
python scripts/bclass_extra_experiments.py --mode train_alibi
```

## 评测 PPL

```bash
export BASE_MODEL_DIR=/path/to/llama3.2-1b-base
export DATA_VALID=./data/wikitext-103/wiki.valid.tokens
export LAOPPE_STATE_DICT=./checkpoints/laoppe_l2_th2048_delta.pt

python scripts/eval_ppl.py
```

等价于：

```bash
python scripts/bclass_extra_experiments.py --mode eval_ppl
```

默认长度：

```text
512, 1024, 2048, 4096
```

可通过环境变量修改：

```bash
export TEST_LENGTHS=512,1024,2048,4096
export NUM_SAMPLES=100
```

## Paired t-test

PPL sample-level 结果生成后，可运行：

```bash
python scripts/bclass_extra_experiments.py --mode ttest
```

输出文件通常为：

```text
bclass_results/ppl_paired_ttest.csv
```

## Needle-in-a-Haystack

```bash
export BASE_MODEL_DIR=/path/to/llama3.2-1b-base
export LAOPPE_STATE_DICT=./checkpoints/laoppe_l2_th2048_delta.pt

python scripts/eval_needle.py
```

等价于：

```bash
python scripts/bclass_extra_experiments.py --mode needle
```

## 8192 PPL / YaRN / LongBench 补充实验

8192 PPL：

```bash
export BASE_MODEL_PATH=/path/to/llama3.2-1b-base
export DATA_VALID=./data/wikitext-103/wiki.valid.tokens
export LAOPPE_STATE_DICT=./checkpoints/laoppe_l2_th2048_delta.pt
export OUT_DIR=./bclass_plus_results

python scripts/bclass_plus_experiments.py ppl8192
```

YaRN Needle 对比：

```bash
python scripts/bclass_plus_experiments.py needle_yarn
```

LongBench 小子集：

```bash
export LONGBENCH_DIR=./LongBench
python scripts/eval_longbench_subset.py
```

或显式指定任务：

```bash
python scripts/bclass_plus_experiments.py longbench \
  --tasks passage_retrieval_en,qasper \
  --max_samples 30 \
  --max_input_tokens 8192
```

生成汇总 Markdown：

```bash
python scripts/make_tables_and_figures.py
```

## 结果文件说明

| 文件 | 说明 |
|---|---|
| `results/ppl_results.csv` | RoPE、PI、NTK、ALiBi、LA-OPPE 等方法在不同长度下的 PPL 汇总 |
| `results/ppl_sample_level.csv` | PPL 的样本级记录，便于做显著性检验 |
| `results/ppl_paired_ttest.csv` | 配对 t-test 结果 |
| `results/needle_results.csv` | Needle-in-a-Haystack 准确率 |
| `results/ppl8192_results.csv` | 8192 长度 PPL 补充结果 |
| `results/needle_yarn_results.csv` | YaRN 等长上下文基线的 Needle 对比 |
| `results/longbench_subset_results.csv` | LongBench 小子集结果 |
| `results/final_experiment_summary.md` | 主实验摘要 |

## 常见问题

### 找不到 `llama_rope_laoppe_score_bias`

请确认你在仓库根目录运行脚本。当前脚本会自动把 `src/` 加入 `sys.path`，也可以手动设置：

```bash
export PYTHONPATH=src:$PYTHONPATH
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="src;$env:PYTHONPATH"
```

### 找不到基础模型

设置：

```bash
export BASE_MODEL_PATH=/path/to/llama3.2-1b-base
export BASE_MODEL_DIR=$BASE_MODEL_PATH
```

并确认目录里有 Hugging Face 格式文件，例如 `config.json`、tokenizer 文件和模型权重文件。

### 找不到 Wikitext 文件

确认路径存在：

```bash
ls ./data/wikitext-103/wiki.train.tokens
ls ./data/wikitext-103/wiki.valid.tokens
```

如果放在其他目录，请设置 `TRAIN_DATA_PATH`、`VALID_DATA_PATH`、`DATA_TRAIN`、`DATA_VALID`。

### CUDA 显存不足

可以尝试：

- 减小 `BLOCK_SIZE`；
- 减小 `NUM_SAMPLES`；
- 使用 `DTYPE=fp16` 或 `DTYPE=bf16`；
- 只跑较短长度；
- 使用更小基础模型；
- 确认没有其他进程占用显存。

### LongBench 分数偏低

LongBench 结果不仅反映长上下文位置建模，也受基础模型指令跟随、生成格式和答案匹配方式影响。建议把 LongBench 当作补充指标，不要只用它判断位置编码方法优劣。

## 开源到 GitHub 的建议步骤

在本地确认仓库结构后：

```bash
git init
git add README.md LICENSE requirements.txt .gitignore configs src scripts results figures docs checkpoints
git status
git commit -m "Release LA-OPPE long-context experiments"
```

在 GitHub 新建空仓库，例如：

```text
LA-OPPE-LongContext
```

然后绑定远端并推送：

```bash
git branch -M main
git remote add origin https://github.com/<your-name>/LA-OPPE-LongContext.git
git push -u origin main
```

推送前建议再检查：

```bash
git status
git ls-files
```

确认没有以下内容：

- 绝对路径旧脚本；
- 终端日志；
- `*.bak`；
- AutoDL 截图；
- 未清洗 jsonl；
- 完整模型权重；
- 第三方数据集原文件。

## 引用

如果你在论文或报告中使用本仓库，可以暂用：

```bibtex
@misc{laoppe_longcontext,
  title        = {Distance-Gated Residual LA-OPPE Score Bias for Long-Context Modeling},
  author       = {LA-OPPE Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-name>/LA-OPPE-LongContext}}
}
```



## License

代码许可证见 `LICENSE`。

注意：本仓库不授权或再分发任何第三方基础模型、数据集或其许可证约束内容。使用 Llama、Wikitext、LongBench 等资源时，请遵守对应项目的许可证和使用条款。
