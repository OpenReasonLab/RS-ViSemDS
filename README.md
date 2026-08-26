# RS-ViSemDS: Visual-Semantic Demonstration Selection for Remote Sensing Scene Classification with Open-Weight Multimodal Large Language Models

<p align="center">
  <a href="assets/fig02_datasets_models_baseline_protocols.pdf">📄 Figure 2: Datasets, Models, and Baseline Protocols</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="assets/fig03_rs_visemds_framework.pdf">🧩 Figure 3: RS-ViSemDS Framework</a>
</p>

本仓库提供论文 **RS-ViSemDS** 的实验复现代码。固定评估划分由 seed 42 生成（每类 100 张测试图像）；论文表格中的性能指标均为十次独立运行的算术平均。正式入口和不可混用的历史入口见 `PAPER_REPRODUCTION.md`。

本代码包仅包含源代码、配置文件、固定数据划分 manifest、测试、运行脚本以及上方两张论文协议图，不包含原始图像、模型权重、RemoteCLIP 缓存、预生成示例、预测结果、指标文件、混淆矩阵、实验结果图表、日志或检查点。运行实验所需的检索文件均由代码在本地生成。

## 🧭 1. 实验范围

### MLLM 基线

- GPT-4o
- Llama-3.2-11B-Vision-Instruct
- Gemma-3-12B
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B
- InternVL3.5-8B
- InternVL3.5-14B

支持以下设置：

- Zero-shot
- Random few-shot
- 全局 RemoteCLIP visual-kNN few-shot

### 传统视觉基线

- ResNet-18
- ResNet-50
- ViT-Tiny
- ViT-Small

支持逐目标、逐类别 RemoteCLIP kNN few-shot 以及 full-data head-only transfer learning。RemoteCLIP 负责检索类平衡示例，四个 ImageNet 预训练骨干保持冻结并只训练分类头。

### RS-ViSemDS

主实验在以下三个开源 MLLM 上运行：

- Gemma-3-12B
- Qwen3-VL-8B
- InternVL3.5-14B

## 🗂️ 2. 代码结构

```text
configs/                         数据集与类别配置
assets/                          论文 Figure 2/3 PDF
data_raw/                        原始数据目录，发布包中为空
examples/                        运行时生成的 Random/kNN 示例目录
manifests/                       论文使用的 seed-42 固定数据划分
strict_fewshot/                  数据、模型、指标、特征与 MLLM 公共组件
RS-ViSemDS/                      RS-ViSemDS 选择、提示、推理与测试代码
build_examples.py                MLLM Random/global-kNN 示例生成器
prepare_eval100_protocol.py      MLLM 示例统一准备入口
run_mllm_paper_runs.py           MLLM 十次正式运行与聚合入口
aggregate_paper_runs.py          四项指标的十次算术平均聚合器
run_*_aid_nwpu_all.py            各 MLLM 的 AID/NWPU 统一入口
run_all_per_class_fewshot.py     四个传统 few-shot 模型统一入口
run_full_data_fixed_eval_all.py  四个 full-data 模型统一入口
verify_package.py                数据划分与纯代码包完整性校验器
```

论文与代码的逐项核查记录见 `MANUSCRIPT_CODE_CONSISTENCY.md`，随机种子和数据集来源见 `SEEDS_AND_DATASETS.md`，GitHub 上传范围与排除项见 `GITHUB_UPLOAD_CHECKLIST.md`。

## 🔒 3. 固定评估协议

| Dataset | Classes | Test images per class | Test total | Support pool |
|---|---:|---:|---:|---:|
| AID | 10 | 100 | 1000 | 2210 |
| NWPU-Urban | 8 | 100 | 800 | 4800 |

- 固定随机种子为 `42`。
- AID manifest：`manifests/aid_eval100_seed42/`。
- NWPU-Urban manifest：`manifests/nwpu_eval100_seed42/`。
- 测试集与 support pool 严格不重叠。
- 测试图像只用于最终评估，不参与示例检索、训练、内部验证、调参或模型选择。
- 旧的每类 24 张测试图像协议及其结果没有放入本包。

> ⚠️ 不要重新随机生成 manifest，否则将不再对应论文使用的固定测试集。

## 🧰 4. 准备数据与环境

数据集下载地址见 `SEEDS_AND_DATASETS.md`。下载并解压后，将类别目录放入：

```text
data_raw/AID_dataset/
data_raw/NWPU-RESISC45/
```

类别目录名称必须与 `configs/*.json` 及 manifest 中的 `source_directory` 一致。

安装公共依赖：

```bash
python -m pip install -r requirements.txt
```

本地 MLLM 还应根据模型官方要求安装兼容版本的 `transformers`、`accelerate` 和相关视觉依赖。FlashAttention2 为可选加速项；未安装时模型可回退到标准 attention，但速度和显存占用可能不同。

## 🔎 5. 生成 MLLM Few-Shot 示例

MLLM 的 Random 和 visual-kNN 中，`k` 表示每个目标图像使用的示例总数，而不是每类示例数。

- Random：每次运行从完整 support pool 中为每个目标重新采样总计 `k` 张图像；十次正式运行使用 seeds 42–51。
- Visual kNN：使用冻结的 RemoteCLIP 图像编码器，从完整 support pool 中检索全局 top-`k` 图像。

数据与 RemoteCLIP 权重就位后运行：

```bash
python prepare_eval100_protocol.py \
  --skip-manifests \
  --shots 1 3 5 10 \
  --strategies random knn \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

`--skip-manifests` 会保留包内的论文固定划分。生成的 CSV 和特征缓存属于运行产物，不包含在发布包中。

## 🤖 6. 运行 MLLM 基线

本地开源模型统一采用冻结参数、`bfloat16`、自动设备映射、贪心解码、`do_sample=False` 和 `max_new_tokens=256`。每张目标图像使用独立的新上下文，输入中不提供文件名、真实标签或图像元数据。

正式实验必须使用十次运行入口。以 InternVL3.5-14B 为例：

```bash
python run_mllm_paper_runs.py \
  --model /path/to/InternVL3.5-14B \
  --model-alias internvl35_14b \
  --backend transformers \
  --datasets aid nwpu_fg_urban \
  --methods zero random knn \
  --shots 3 --runs 10 --resume
```

`run_*_aid_nwpu_all.py` 是模型专用的单次运行/环境调试入口，不直接产生论文十次均值。支持的模型专用入口包括：

```text
run_llama32_11b_aid_nwpu_all.py
run_gemma3_12b_aid_nwpu_all.py
run_qwen25vl_7b_aid_nwpu_all.py
run_qwen3vl_8b_aid_nwpu_all.py
run_internvl35_8b_aid_nwpu_all.py
```

GPT-4o 默认使用官方 OpenAI API：

```bash
export OPENAI_API_KEY="your-key"
python run_mllm_paper_runs.py \
  --model gpt-4o \
  --model-alias gpt4o \
  --backend api \
  --datasets aid nwpu_fg_urban \
  --methods zero random knn \
  --shots 3 --runs 10 --resume
```

Zero-shot、Random few-shot 和 visual-kNN few-shot 不使用类别描述或边界规则。解析器先读取规定 JSON；若格式偏离但输出中只出现一个候选类别，仍按该唯一类别计；否则视为无效输出，不重试并按错误分类计。

MLLM 入口支持 `--limit 5` 做小规模检查，支持 `--dry-run` 查看命令。正式实验不要设置 `--limit`。

AutoDL 上可用以下脚本准备模型环境和做单次检查；论文表格仍由 `run_mllm_paper_runs.py` 产生：

```bash
MODEL_ALIAS=internvl35_14b \
MLLM_MODEL_PATH=/root/autodl-tmp/models/InternVL3.5-14B \
PREPARE_EXAMPLES=1 \
REMOTECLIP_CHECKPOINT=/path/to/RemoteCLIP-ViT-B-32.pt \
bash ./run_open_mllm_eval100_autodl.sh
```

## 🖼️ 7. 运行传统 Few-Shot 基线

对于每个目标图像，代码使用冻结的 RemoteCLIP 从 support pool 的每个类别分别检索最相似的 `k_cls` 张图像。相同的目标专属类平衡训练集供四个冻结 ImageNet 骨干使用，大小为 `k_cls × C`。

训练设置为：只优化新分类头，10 epochs，batch size 16，Adam，学习率 0.001，交叉熵损失，输入尺寸 224×224。正式入口运行十个独立训练种子（默认 42–51）并分别聚合 Accuracy、macro Precision、macro Recall 和 macro F1。

首次运行时生成共享的 RemoteCLIP 检索文件并完成十次实验：

```bash
python run_all_per_class_fewshot.py \
  --datasets aid nwpu \
  --shots 1 3 5 10 \
  --models resnet18 resnet50 vit_tiny vit_small
```

只生成逐类 RemoteCLIP kNN 文件：

```bash
python run_all_per_class_fewshot.py \
  --datasets aid nwpu \
  --models resnet18 resnet50 vit_tiny vit_small \
  --examples-only
```

仅当本机已经生成相同 manifest、模型和 shot 对应的检索文件时，才可使用 `--skip-examples`。

## ⚙️ 8. 运行 Full-Data 基线

Full-data 的开发划分固定使用 seed 42；最终完整 support 训练默认使用 seeds 42–51。每个 ImageNet 预训练骨干保持冻结，只训练新初始化的分类头：

1. 将 support pool 按 90%/10% 划分为内部训练集和验证集，用于确定训练配置。
2. 重新初始化分类头，在完整 support pool 上训练固定 10 epochs。
3. 每个种子只使用第 10 轮模型在固定测试集评估一次，再对十次结果取算术平均。

```bash
python run_full_data_fixed_eval_all.py
```

测试集始终不参与内部验证、学习率调整或模型选择。更完整的实现参数及论文披露差异见 `MANUSCRIPT_CODE_CONSISTENCY.md`。

## ✨ 9. 运行 RS-ViSemDS

论文主实验使用以下固定配置：

- RemoteCLIP 图像与文本编码器。
- 每类候选数 `r=3`。
- 最终示例总数 `k=3`。
- 评分权重 `(alpha, beta, gamma)=(0.6, 0.2, 0.2)`。
- 论文附录两阶段提示模式 `manuscript_v1`。

```bash
python RS-ViSemDS/run_rs_visemds_all.py \
  --datasets aid nwpu_fg_urban \
  --models gemma3_12b qwen3vl_8b internvl35_14b \
  --model-path gemma3_12b=/path/to/gemma-3-12b-it \
  --model-path qwen3vl_8b=/path/to/Qwen3-VL-8B \
  --model-path internvl35_14b=/path/to/InternVL3.5-14B \
  --r 3 --k 3 \
  --alpha 0.6 --beta 0.2 --gamma 0.2 \
  --prompt-mode manuscript_v1 \
  --runs 10 \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

算法、评分、类别文本、提示构造、时间统计和单模型运行方式见 `RS-ViSemDS/README.md`。其他 prompt mode 和权重设置仅用于辅助分析或消融，不应与论文主实验混用。

## ✅ 10. 校验代码包

原始数据尚未放入时：

```bash
python verify_package.py
python -m unittest discover -s tests -v
python -m unittest discover -s RS-ViSemDS/tests -v
```

放入原始数据后：

```bash
python verify_package.py --allow-data
```

校验器会检查固定测试集的每类数量、support/test 泄漏、必要入口、文件哈希，以及包内是否意外包含结果、日志、缓存、权重、压缩包或预生成示例。
