# MLLM 快速开始（本地 Qwen3-VL 评测 MathVista testmini）

本文档用于在本机快速跑通以下脚本：

- `examples/mllm/eval_mathvista_testmini_qwen3vl_local.py`

## 1. 前置条件

- 已安装并可用 `conda`
- 可用环境：`base`（或你自己的 Python 环境）
- 本地模型目录存在：`/gz-data/qwen3vl_2b`
- 测试数据文件存在：`/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet`

vllm >= 0.11.0
verl >= 0.6.1


可先做一次快速检查：

```bash
ls -lah /gz-data/qwen3vl_2b
ls -lah /gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet
```

## 2. 一键运行完整评测

```bash
cd /gz-data/AlphaApollo-Visual

conda run -n base python -u examples/mllm/eval_mathvista_testmini_qwen3vl_local.py \
	--model-path /gz-data/qwen3vl_2b \
	--data-file /gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
	--max-samples -1000 \
	--max-new-tokens 196 \
	--output-dir logs/mathvista_testmini_eval_full
```

说明：

- `--max-samples -1000` 表示评测全部样本
- 如果你只想先验证流程，可改成 `--max-samples 10`

## 3. 常用参数

- `--model-path`：本地模型路径
- `--data-file`：待评测 parquet 文件
- `--max-samples`：评测样本数；`-1000` 为全量
- `--max-new-tokens`：每条样本最大生成长度
- `--do-sample`：开启采样生成（默认关闭）
- `--temperature`：采样温度（仅在 `--do-sample` 时生效）
- `--top-p`：采样 top-p（仅在 `--do-sample` 时生效）
- `--output-dir`：输出目录
- `--predictions-file`：逐样本明细文件名，默认 `predictions.jsonl`
- `--metrics-file`：聚合指标文件名，默认 `metrics.json`
- `--case-study-file`：案例分析文件名，默认 `case_study.md`

## 4. 输出结果说明

运行成功后，`--output-dir` 下会生成：

- `predictions.jsonl`：每个样本一行，含 `gold_answer`、`prediction`、`correct`、`response` 等字段
- `metrics.json`：整体准确率、按题型统计、按 task/context 分组统计
- `case_study.md`：成功和失败案例摘要，便于快速排查

你可以这样快速查看指标：

```bash
cat /gz-data/AlphaApollo-Visual/logs/mathvista_testmini_eval_full/metrics.json
```

## 5. 建议的最小验证流程

先跑小样本（10 条）确认环境正常，再跑全量：

```bash
cd /gz-data/AlphaApollo-Visual

conda run -n base python -u examples/mllm/eval_mathvista_testmini_qwen3vl_local.py \
	--model-path /gz-data/qwen3vl_2b \
	--data-file /gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
	--max-samples 10 \
	--max-new-tokens 196 \
	--output-dir logs/mathvista_testmini_eval_smoke
```

## 6. 常见问题

- 报错找不到模型：确认 `--model-path` 指向本地模型目录（包含 `config.json`、`tokenizer.json`、`model.safetensors` 等文件）
- 显存不足：先降低 `--max-new-tokens`，并改用小样本 `--max-samples 10`
- 结果目录已存在：可直接复用目录，或更换 `--output-dir` 避免覆盖
- 速度较慢：建议先做 smoke 跑通后再做全量评测

## 7. 相关脚本

- `examples/mllm/test_qwen3vl_local_smoke.py`：检查本地模型 + 数据集读取流程是否正常
- `examples/mllm/test_qwen3vl_local_mm_infer_10.py`：对前 N 条样本做多模态推理，便于人工检查输出