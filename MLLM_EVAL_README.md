# 多模态评测功能说明（eval_vllm.py）

本文档说明 `examples/mllm/eval_vllm.py` 新增的多模态评测功能，包括对框架的核心改动、运行教程、复现结果汇总，以及过程中遇到的问题与解决方案。

---

## 一、核心改动点

### 1. `rl_dataset.py` — 多模态数据加载支持

文件路径：`alphaapollo/core/generation/verl/utils/dataset/rl_dataset.py`

**改动背景**：原始 `RLHFDataset` 仅支持纯文本 prompt，无法处理图像/视频输入。

**主要改动**：

#### 1.1 新增 `processor` 参数

```python
def __init__(self, data_files, tokenizer, config, processor: Optional[ProcessorMixin] = None):
```

通过 `ProcessorMixin`（如 `AutoProcessor`）统一处理图文输入，兼容 Qwen2-VL / Qwen3-VL 等多模态模型。

#### 1.2 `_processor_supports_multimodal_kwargs()` — 多模态能力探测

通过检查 processor 是否有 `image_processor` 属性，或其 `__call__` 签名是否包含 `images`/`videos` 参数，动态判断是否走多模态路径，避免对纯文本 processor 的误用。

#### 1.3 `_pick_image_items()` — 图像路径解析与回退机制

- 优先使用 `image_key`（本地路径）
- 若路径不存在，自动回退到 `decoded_image_key`（bytes/base64/data URI）
- 支持相对路径解析（相对于 parquet 文件目录或 `image_root`）

#### 1.4 `__getitem__` — 多模态/纯文本双路径

多模态路径下额外输出：
- `multi_modal_data`：原始图像/视频数据（用于 env 传递）
- `multi_modal_inputs`：processor 输出的张量（`pixel_values`、`image_grid_thw` 等）

纯文本路径保持原有行为，向后兼容。

#### 1.5 Qwen2-VL RoPE position_ids 适配

Qwen2-VL/Qwen3-VL 使用 3D RoPE，需要根据 `image_grid_thw` 调用 `get_rope_index()` 计算正确的 position_ids，否则图像 token 位置编码错误。

---

### 2. `eval_vllm.py` — 基于 vLLM + Ray 的并行评测脚本

**改动背景**：`eval_baseline.py` 使用 HuggingFace `model.generate()` 串行推理，速度慢，无法利用 vLLM 的连续批处理和 PagedAttention。

**核心设计**：

#### 2.1 推理后端替换：vLLM + Ray Worker Group

通过 AlphaApollo 的 `RayWorkerGroup` 管理 vLLM worker，支持 tensor parallel 和多 GPU 部署：

```python
ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
resource_pool = RayResourcePool(process_on_nodes=[n_gpus], max_colocate_count=1)
worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda")
worker_group.init_model()
```

#### 2.2 `TrajectoryCollector.multi_turn_loop` 作为推理入口

复用 RL 训练的 rollout 接口做推理，`env.max_steps=1` 退化为单步生成，无需修改 rollout 核心逻辑：

```python
traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
gen_batch_output = traj_collector.multi_turn_loop(gen_batch=gen_batch, actor_rollout_wg=worker_group, envs=envs, is_train=False)
```

#### 2.3 `OneStepMathVistaEnvManager` — 最小化 Env 适配器

将 MathVista 数据集包装成 rollout_loop 期望的 env 接口（`reset` / `step` / `success_evaluator`），无需修改 `TrajectoryCollector`。`step` 直接返回 `done=True`，`success_evaluator` 返回占位零数组。

#### 2.4 ThreadPoolExecutor 并行数据预处理

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=args.data_num_workers) as executor:
    samples = list(executor.map(dataset.__getitem__, sample_indices))
```

`RLHFDataset.__getitem__` 包含图像解码和 processor 调用（CPU 密集），用线程池并行预处理一个 batch 内的样本，避免 GPU 等待数据。

> 未换成 Ray 并行的原因：`__getitem__` 依赖 `self.tokenizer`/`self.processor` 实例状态，序列化开销大；与现有 `RayWorkerGroup` 混用资源调度复杂；ThreadPoolExecutor 已解决主要瓶颈。

#### 2.5 显存与 Ray 对象存储主动释放

每个 batch 结束后主动释放，防止长时间评测中 VRAM 和 Ray 对象存储累积溢出：

```python
del gen_batch_output, gen_batch, batch, batch_dict, samples, rows
torch.cuda.empty_cache()
gc.collect()
unreferenced = [oid for oid, cnt in ray._private.state.state.get_all_reference_counts().items() if cnt == 0]
if unreferenced:
    ray._private.internal_api.free(unreferenced, local_only=True)
```

#### 2.6 `build_config()` — 从 generation.yaml 构建配置

复用框架的 `generation.yaml` 基础配置，通过命令行参数覆盖关键字段，保持与训练配置的一致性。

---

## 二、运行教程

### 环境安装

环境：Python 3.12.7 / PyTorch 2.8.0 / CUDA 12.8，完整依赖见 `environment.yml`。

```bash
# 从 environment.yml 一键还原（推荐）
conda env create -f environment.yml -n alphaapollo-mllm
conda activate alphaapollo-mllm
```

或手动安装核心包：

```bash
conda create -n alphaapollo-mllm python=3.12.7 -y
conda activate alphaapollo-mllm
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.11.0 verl==0.6.1 transformers==4.57.0 ray==2.54.0 \
    accelerate==1.13.0 datasets==4.7.0 omegaconf==2.3.0 flash-attn==2.8.3 \
    qwen-vl-utils==0.0.14 tensordict==0.10.0
```

### 前置条件

```
vllm >= 0.11.0
verl >= 0.6.1
ray（随 verl 安装）
```

确认资源：

```bash
ls /gz-data/qwen3vl_4b
ls /gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet
```

### 快速验证（10 条样本）

```bash
cd /gz-data/AlphaApollo-Visual
MAX_SAMPLES=10 MODEL_PATH=/gz-data/qwen3vl_4b bash examples/mllm/run_eval_vllm.sh
```

### 完整评测（1000 条，单卡）

```bash
cd /gz-data/AlphaApollo-Visual

MODEL_PATH=/gz-data/qwen3vl_4b \
DATA_FILE=/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
OUTPUT_DIR=logs/vllm_qwen3vl_4b_testmini_full_parallel/max_new_token_1024_version_bs1 \
MAX_SAMPLES=1000 \
BATCH_SIZE=1 \
MAX_NEW_TOKENS=1024 \
N_GPUS_PER_NODE=1 \
TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.6 \
FREE_CACHE_ENGINE=true \
bash examples/mllm/run_eval_vllm.sh
```

或直接调用 Python：

```bash
cd /gz-data/AlphaApollo-Visual

python3 -u examples/mllm/eval_vllm.py \
    --model-path /gz-data/qwen3vl_4b \
    --data-file /gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
    --output-dir logs/vllm_qwen3vl_4b_testmini_full_parallel/max_new_token_1024_version_bs1 \
    --max-samples 1000 \
    --batch-size 1 \
    --max-new-tokens 1024 \
    --n-gpus-per-node 1 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8172 \
    --free-cache-engine
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `/gz-data/qwen3vl_2b` | 本地模型目录 |
| `--data-file` | testmini parquet | 评测数据集 |
| `--batch-size` | 8 | 每批样本数 |
| `--max-new-tokens` | 1024 | 最大生成长度 |
| `--max-samples` | 1000 | 评测样本数，≤0 为全量 |
| `--n-gpus-per-node` | 1 | rollout worker 使用的 GPU 数 |
| `--tensor-parallel-size` | 1 | vLLM tensor parallel |
| `--gpu-memory-utilization` | 0.9 | vLLM 显存占用比例 |
| `--free-cache-engine` | false | 每 batch 后释放 KV cache engine |
| `--data-num-workers` | 4 | 数据预处理线程数 |
| `--ray-num-cpus` | 8 | Ray 初始化 CPU 资源 |
| `--max-model-len` | 8192 | vLLM 最大序列长度 |

### 输出文件

运行完成后 `--output-dir` 下生成：

- `predictions.jsonl`：每条样本的完整推理记录（query、response、gold_answer、correct 等）
- `metrics.json`：整体准确率 + 按题型/任务/上下文分组统计
- `case_study.md`：成功和失败案例摘要

```bash
cat logs/vllm_qwen3vl_4b_testmini_full_parallel/max_new_token_1024_version_bs1/metrics.json
```

---

## 三、复现结果汇总

详细结果见：`examples/mllm/results_summary.md`

### 总体准确率

| 模型 | 推理方式 | max_new_tokens | 总体 | 多选题 | 填空题 |
|------|---------|---------------|------|-------|-------|
| Qwen3-VL-2B (official) | — | — | 61.3% | — | — |
| Qwen3-VL-2B | baseline (HF generate) | 196 | 42.7% | 48.0% | 36.5% |
| Qwen3-VL-2B | baseline | 512 | 54.5% | 61.5% | 46.3% |
| Qwen3-VL-2B | baseline | 1024 | 57.9% | 66.5% | 47.8% |
| Qwen3-VL-2B | vllm | 512 | 54.1% | 61.9% | 45.0% |
| Qwen3-VL-2B | vllm | 1024 | 57.0% | 66.1% | 46.3% |
| Qwen3-VL-2B | vllm | 2048 | 57.3% | 66.7% | 46.3% |
| Qwen3-VL-4B (official) | — | — | 73.7% | — | — |
| **Qwen3-VL-4B** | **vllm** | **1024** | **67.5%** | **75.6%** | **58.0%** |

> Qwen3-VL-4B vllm 复现（675/1000），与官方 73.7% 的差距主要来自答案提取策略和 thinking token 处理，非推理框架问题。

### 按任务类型（Qwen3-VL-4B，vllm，max_new_tokens=1024）

| 任务 | 样本数 | 准确率 |
|------|--------|--------|
| figure question answering | 269 | 71.7% |
| geometry problem solving | 208 | 68.8% |
| math word problem | 186 | 72.0% |
| visual question answering | 179 | 52.0% |
| textbook question answering | 158 | 70.9% |

### 按上下文类型（Qwen3-VL-4B，vllm，max_new_tokens=1024）

| 上下文 | 样本数 | 准确率 |
|--------|--------|--------|
| table | 70 | 92.9% |
| function plot | 62 | 82.3% |
| bar chart | 119 | 78.2% |
| geometry diagram | 216 | 69.9% |
| natural image | 109 | **35.8%**（最低） |

---

## 四、过程中遇到的问题与解决方案

### 问题 1：Qwen3-VL config.json 中 model_type 不被 vLLM 识别

**现象**：vLLM 加载 Qwen3-VL-2B 时报错，找不到对应的模型实现。

**原因**：Qwen3-VL 的 `config.json` 中 `model_type` 为 `qwen3_vl`，而当时版本的 vLLM 只注册了 `qwen2_vl`。

**解决方案**：临时修改模型目录下的 `config.json`：

```bash
cd /gz-data/qwen3vl_2b
sed -i 's/Qwen3VLForConditionalGeneration/Qwen2VLForConditionalGeneration/g' config.json
sed -i 's/"model_type": "qwen3_vl"/"model_type": "qwen2_vl"/g' config.json
```

> 此修改已在 `run_eval_vllm.sh` 中注释保留，升级 vLLM 版本后可移除。

---

### 问题 2：Ray worker 初始化时 assertion error（资源未就绪）

**现象**：`worker_group.init_model()` 偶发 assertion error，尤其在刚启动 Ray 后立即调用时。

**原因**：Ray worker 进程启动是异步的，`ray.init()` 返回后 worker 可能尚未完全就绪。

**解决方案**：在 `init_model()` 前加入固定延迟：

```python
import time
time.sleep(10)
worker_group.init_model()
```

---

### 问题 3：RayResourcePool 默认 max_colocate_count=10 导致 CPU bundle 过大

**现象**：Ray placement group 申请资源时失败，提示 CPU 资源不足。

**原因**：`RayResourcePool` 默认 `max_colocate_count=10`，会为每个 bundle 申请 10 个 CPU slot，超出实际可用 CPU 数。

**解决方案**：显式设置 `max_colocate_count=1`：

```python
resource_pool = RayResourcePool(
    process_on_nodes=[config.trainer.n_gpus_per_node],
    max_colocate_count=1,
)
```

---

### 问题 4：多 batch 评测中 VRAM 持续增长直至 OOM

**现象**：评测前几个 batch 正常，之后 VRAM 占用持续上升，最终 OOM。

**原因**：`gen_batch_output` 等大张量未及时释放；Ray plasma store 中引用计数降为 0 的对象不会立即释放内存。

**解决方案**：每个 batch 结束后主动清理：

```python
del gen_batch_output, gen_batch, batch, batch_dict, samples, rows
torch.cuda.empty_cache()
gc.collect()
unreferenced = [oid for oid, cnt in ray._private.state.state.get_all_reference_counts().items() if cnt == 0]
if unreferenced:
    ray._private.internal_api.free(unreferenced, local_only=True)
```

同时启用 `--free-cache-engine` 让 vLLM 在每次推理后释放 KV cache engine。

---

### 问题 5：`second_per_grid_ts` 字段导致 DataProto 序列化报错

**现象**：vLLM 推理时报错，提示 `second_per_grid_ts` 字段类型不兼容。

**原因**：Qwen3-VL processor 输出的 `second_per_grid_ts` 是视频时序字段，在图像评测中为空或类型不一致，无法被 `DataProto` 正确序列化。

**解决方案**：在 `rl_dataset.py` 的 `__getitem__` 中主动 pop 掉该字段：

```python
if "second_per_grid_ts" in model_inputs:
    model_inputs.pop("second_per_grid_ts")
row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
```

---

### 问题 6：`data_source` / `raw_prompt` 字段缺失导致 `preprocess_single_sample` 报错

**现象**：`TrajectoryCollector.multi_turn_loop` 内部调用 `preprocess_single_sample` 时 KeyError。

**原因**：MathVista parquet 数据集中没有 `data_source` 和 `raw_prompt` 字段，但 rollout 流程默认这两个字段存在。

**解决方案**：在构建 batch 前手动补充默认值：

```python
for row, sample in zip(rows, samples):
    if "data_source" not in sample:
        sample["data_source"] = "mathvista"
    if "raw_prompt" not in sample:
        sample["raw_prompt"] = [{"role": "user", "content": query}]
```

---

### 问题 7：图像路径在 parquet 中为相对路径，加载失败

**现象**：`process_image()` 报 FileNotFoundError，图像路径如 `images/xxx.png` 无法找到。

**原因**：MathVista parquet 中 `image` 字段存储的是相对路径，但运行目录与数据集目录不一致。

**解决方案**：`rl_dataset.py` 中的 `_resolve_image_item()` 按优先级尝试多个候选根目录（`image_root` 配置 → parquet 文件所在目录 → 其父目录），找到第一个存在的路径后返回。若所有路径均不存在，则回退到 `decoded_image` 字段中的 bytes/base64 数据。
