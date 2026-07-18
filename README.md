# Eval100 Remote-Sensing Classification Reproducibility Package

本目录整理了论文中“每类固定 100 张评估图像”协议使用的代码。它只包含代码、配置、固定划分 manifest 和可复现的检索索引，不包含原始图像、模型权重、运行结果、日志或缓存。

## 1. 实验范围

MLLM（0-shot、Random few-shot、RemoteCLIP kNN few-shot）：

- GPT-4o
- Llama-3.2-11B-Vision-Instruct
- Gemma-3-12B
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B
- InternVL3.5-8B
- InternVL3.5-14B

传统视觉模型：

- ResNet-18
- ResNet-50
- ViT-Tiny
- ViT-Small

另外包含完整的 `RS-ViSemDS/` 实现、测试和 AutoDL 脚本。

## 2. 固定评估协议

- AID：10 个类别，每类 100 张评估图像，共 1000 张；support pool 共 2210 张。
- NWPU-Urban：8 个类别，每类 100 张评估图像，共 800 张；support pool 共 4800 张。
- 固定划分位于 `manifests/aid_eval100_seed42` 和 `manifests/nwpu_eval100_seed42`。
- 评估图像与 support pool 严格不重叠。
- 旧的每类 24 张评估协议及旧 manifest 没有放入本包。

完整种子记录和数据集链接见 `SEEDS_AND_DATASETS.md`。

## 3. 放置数据

当前两个目录故意保持为空：

```text
data_raw/AID_dataset/
data_raw/NWPU-RESISC45/
```

下载并解压后，把每个类别目录直接放到相应目录中。类别目录名称必须与 `configs/*.json` 和 manifest 中的 `source_directory` 一致。不要重新随机生成 manifest，否则将不再是论文使用的 seed 42 固定评估集。

## 4. 安装环境

```bash
python -m pip install -r requirements.txt
```

本地 Transformers MLLM 还应按模型官方要求安装匹配版本的 `transformers`、`accelerate` 和可选的 FlashAttention2。GPT-4o 使用兼容 OpenAI Chat Completions 的 API。

## 5. 生成 MLLM few-shot 检索文件

本包保留了固定 manifest，但没有复制本机尚未生成的 MLLM Random/global-kNN CSV。数据和 RemoteCLIP 权重就位后执行：

```bash
python prepare_eval100_protocol.py \
  --skip-manifests \
  --shots 1 3 5 10 \
  --strategies random knn \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

`--skip-manifests` 很重要：它会复用包内 seed 42 的固定评估划分。传统模型所用的 per-class RemoteCLIP kNN 索引已经放在 `examples/*_remoteclip_knn_per_class/`。

## 6. 运行 MLLM

本地开源模型入口的形式相同，下面以 InternVL3.5-14B 为例：

```bash
python run_internvl35_14b_aid_nwpu_all.py \
  --model /path/to/InternVL3.5-14B \
  --datasets aid nwpu_fg_urban \
  --shots 1 3 5 10
```

其余入口：

```text
run_llama32_11b_aid_nwpu_all.py
run_gemma3_12b_aid_nwpu_all.py
run_qwen25vl_7b_aid_nwpu_all.py
run_qwen3vl_8b_aid_nwpu_all.py
run_internvl35_8b_aid_nwpu_all.py
```

GPT-4o：

```bash
export OPENAI_API_KEY="your-key"
python run_gpt4o_aid_nwpu_all.py \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o \
  --datasets aid nwpu_fg_urban \
  --shots 3 5 10
```

所有入口都支持 `--limit 5` 做小规模检查和 `--dry-run` 查看将执行的命令。正式结果不要设置 `--limit`。

## 7. 运行传统模型

每类 kNN few-shot（默认 1/3/5/10-shot、四个模型、训练 10 轮）：

```bash
python run_all_per_class_fewshot.py --skip-examples
```

如需重建 per-class RemoteCLIP 检索索引：

```bash
python run_all_per_class_fewshot.py \
  --examples-only \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

固定评估集 full-data（四个模型，seed 42/43/44，最大 10 轮，并按当前代码只报告第 10 轮设置）：

```bash
python run_full_data_fixed_eval_all.py
```

## 8. 运行 RS-ViSemDS

```bash
python RS-ViSemDS/run_rs_visemds_all.py \
  --datasets aid nwpu_fg_urban \
  --models gemma3_12b qwen3vl_8b internvl35_14b \
  --model-path gemma3_12b=/path/to/gemma-3-12b-it \
  --model-path qwen3vl_8b=/path/to/Qwen3-VL-8B \
  --model-path internvl35_14b=/path/to/InternVL3.5-14B \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

RS-ViSemDS 的算法、提示构造、检索评分和单模型运行方式见 `RS-ViSemDS/README.md`。

## 9. 校验代码包

数据尚未放入时：

```bash
python verify_package.py
python -m unittest discover -s RS-ViSemDS/tests -v
```

数据放入后使用：

```bash
python verify_package.py --allow-data
```

