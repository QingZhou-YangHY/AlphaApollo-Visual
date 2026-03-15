#!/usr/bin/env python3
"""MathVista testmini evaluation via AlphaApollo rollout_loop + vLLM backend.

Usage:
  cd /gz-data/AlphaApollo-Visual
  python examples/mllm/eval_mathvista_testmini_qwen3vl_rollout_vllm.py \
      --model-path /gz-data/qwen3vl_2b \
      --data-file /gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet

为什么没换成ray实现并行预处理?

现在整个数据流是基于 HuggingFace datasets + RLHFDataset，
dataset[i] 的访问接口、collate_fn、DataProto 的构建都要跟着改

preprocess_single_sample 依赖 self.tokenizer、self.processor 这些实例状态，传给 Ray remote 函数需要序列化，有额外开销

Ray 的 actor/worker 和现有的 RayWorkerGroup 混用，资源调度会更复杂

现在用 ThreadPoolExecutor 已经解决了主要瓶颈，如果有跨节点分布式数据预处理的需求再换
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import concurrent.futures

import numpy as np
import ray
import torch
import os
from vllm.model_executor.models import ModelRegistry
from omegaconf import OmegaConf
from tqdm import tqdm


def ensure_repo_importable() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    # Ensure local verl is prioritized over installed one
    verl_path = str(repo_root / "alphaapollo" / "core" / "generation")
    if verl_path not in sys.path:
        sys.path.insert(0, verl_path)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_final_segment(response: str) -> str:
    text = normalize_text(response)
    if not text:
        return ""

    marker_patterns = [
        r"(?:final\s*answer|answer|答案|最终答案)\s*[:：]\s*(.+)$",
        r"(?:therefore|thus|so)\s*,?\s*the\s*answer\s*is\s*(.+)$",
    ]
    for pat in marker_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    chunks = re.split(r"[\n。.!?；;]", text)
    chunks = [c.strip() for c in chunks if c and c.strip()]
    return chunks[-1] if chunks else text


def extract_multi_choice_letter(text: str, num_choices: int) -> str | None:
    if not text:
        return None

    valid_letters = [chr(ord("A") + i) for i in range(num_choices)]

    patterns = [
        r"\(([A-Za-z])\)",
        r"\boption\s*([A-Za-z])\b",
        r"\bchoice\s*([A-Za-z])\b",
        r"\banswer\s*(?:is|:)\s*([A-Za-z])\b",
        r"\b([A-Za-z])\b",
    ]
    for pat in patterns:
        letters = re.findall(pat, text, flags=re.IGNORECASE)
        for raw in reversed(letters):
            up = raw.upper()
            if up in valid_letters:
                return up
    return None


def get_most_similar(candidate: str, choices: list[str]) -> str:
    if not choices:
        return ""
    sims = [
        difflib.SequenceMatcher(
            a=normalize_text(candidate).lower(),
            b=normalize_text(c).lower(),
        ).ratio()
        for c in choices
    ]
    idx = max(range(len(choices)), key=lambda i: sims[i])
    return choices[idx]


def normalize_extraction(
    extraction: str,
    choices: list[str] | None,
    question_type: str,
    answer_type: str,
    precision: Any,
) -> str | None:
    choices = choices or []
    extraction = normalize_text(extraction)

    if question_type == "multi_choice":
        if not extraction:
            return None

        letter = extract_multi_choice_letter(extraction, len(choices))
        if letter is not None:
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choices):
                return normalize_text(choices[idx])

        for c in choices:
            if normalize_text(c).lower() == extraction.lower():
                return normalize_text(c)
        return normalize_text(get_most_similar(extraction, choices)) if choices else extraction

    if answer_type == "integer":
        num = find_last_number(extraction)
        if num is None:
            return None
        try:
            return str(int(float(num)))
        except Exception:
            return None

    if answer_type == "float":
        num = find_last_number(extraction)
        if num is None:
            return None
        try:
            p = int(float(precision)) if precision is not None else 2
            return str(round(float(num), p))
        except Exception:
            return None

    if answer_type == "list":
        m = re.search(r"\[[^\]]*\]", extraction)
        candidate = m.group(0) if m else extraction
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, list):
                return str(obj)
        except Exception:
            pass
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", extraction)
        if nums:
            return str([float(x) if "." in x else int(x) for x in nums])
        return None

    return extraction if extraction else None


def safe_equal(prediction: str | None, answer: Any) -> bool:
    if prediction is None:
        return False
    return normalize_text(prediction) == normalize_text(answer)


def compute_group_accuracy(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        if key == "task":
            value = normalize_text((r.get("metadata") or {}).get("task", "unknown")) or "unknown"
        elif key == "context":
            value = normalize_text((r.get("metadata") or {}).get("context", "unknown")) or "unknown"
        else:
            value = normalize_text(r.get(key, "unknown")) or "unknown"
        counts[value][1] += 1
        if r.get("correct", False):
            counts[value][0] += 1

    for value, (correct, total) in sorted(counts.items(), key=lambda x: (-x[1][1], x[0])):
        grouped[value] = {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else 0.0,
        }
    return grouped


def save_case_study(records: list[dict[str, Any]], out_file: Path, num_success: int = 2, num_fail: int = 2) -> None:
    success = [r for r in records if r.get("correct")][:num_success]
    fail = [r for r in records if not r.get("correct")][:num_fail]

    lines: list[str] = []
    lines.append("# MathVista Case Study")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    def _emit_block(title: str, arr: list[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not arr:
            lines.append("(no samples)")
            lines.append("")
            return
        for i, r in enumerate(arr, start=1):
            lines.append(f"### {title} #{i} | pid={r.get('pid')}")
            lines.append(f"- question_type: {r.get('question_type')}")
            lines.append(f"- answer_type: {r.get('answer_type')}")
            md = r.get("metadata") or {}
            lines.append(f"- task: {md.get('task', 'unknown')}")
            lines.append(f"- context: {md.get('context', 'unknown')}")
            lines.append(f"- gold_answer: {r.get('gold_answer')}")
            lines.append(f"- extraction: {r.get('extraction')}")
            lines.append(f"- normalized_prediction: {r.get('prediction')}")
            lines.append(f"- correct: {r.get('correct')}")
            lines.append("")
            lines.append("[Query]")
            lines.append(str(r.get("query", "")))
            lines.append("")
            lines.append("[Model Response / Full Reasoning]")
            lines.append(str(r.get("response", "")))
            lines.append("")

    _emit_block("Success Cases", success)
    _emit_block("Failure Cases", fail)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")


class OneStepMathVistaEnvManager:
    """A minimal env manager that adapts dataset samples to rollout_loop's API.

    - reset: returns current query and image as observation.
    - step: one-step episode, marks all envs done.
    - success_evaluator: returns placeholder success array.
    """

    def __init__(self):
        self._batch_size = 0

    @staticmethod
    def _normalize_kwargs(kwargs: Any) -> list[dict[str, Any]]:
        if kwargs is None:
            return []
        if isinstance(kwargs, np.ndarray):
            return [dict(x) for x in kwargs.tolist()]
        if isinstance(kwargs, list):
            return [dict(x) for x in kwargs]
        raise TypeError(f"Unsupported env kwargs type: {type(kwargs)}")

    def reset(self, kwargs=None):
        sample_kwargs = self._normalize_kwargs(kwargs)
        self._batch_size = len(sample_kwargs)

        obs_text = [normalize_text(x.get("query", "")) for x in sample_kwargs]
        obs_images = [x.get("image", None) for x in sample_kwargs]
        has_image = any(img is not None for img in obs_images)

        infos = [{"won": 0.0, "is_action_valid": True, "tool_calling": False} for _ in range(self._batch_size)]
        observations = {
            "text": obs_text,
            "image": obs_images if has_image else None,
            "anchor": [None for _ in range(self._batch_size)],
        }
        return observations, infos

    def step(self, text_actions):
        batch_size = len(text_actions)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.ones(batch_size, dtype=bool)
        infos = [
            {
                "won": 0.0,
                "is_action_valid": True,
                "tool_calling": False,
            }
            for _ in range(batch_size)
        ]
        next_obs = {
            "text": ["" for _ in range(batch_size)],
            "image": None,
            "anchor": [None for _ in range(batch_size)],
        }
        return next_obs, rewards, dones, infos

    def success_evaluator(self, *args, **kwargs):
        # The internal AlphaApollo caller expects 'total_batch_list' or 'total_infos' to be in kwargs.
        batch_size = len(kwargs.get("total_batch_list", [])) or len(kwargs.get("total_infos", []))
        # Ensure we return a dict with 'success_rate' mapping to a numpy array of correct size.
        return {"success_rate": np.zeros(batch_size, dtype=np.float32)}

    def close(self):
        return None


def build_config(args: argparse.Namespace):
    base_cfg_path = Path(__file__).resolve().parents[2] / "alphaapollo" / "core" / "generation" / "verl" / "trainer" / "config" / "generation.yaml"
    config = OmegaConf.load(str(base_cfg_path))

    config.model.path = args.model_path
    config.model.trust_remote_code = True

    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    config.ray_init.num_cpus = args.ray_num_cpus

    config.rollout.name = "vllm"
    config.rollout.mode = "sync"
    config.rollout.tensor_model_parallel_size = args.tensor_parallel_size
    config.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    config.rollout.prompt_length = args.max_prompt_length
    config.rollout.response_length = args.max_new_tokens
    config.rollout.temperature = args.temperature
    config.rollout.top_p = args.top_p
    config.rollout.top_k = args.top_k
    config.rollout.n = 1
    config.rollout.free_cache_engine = args.free_cache_engine
    config.rollout.max_model_len = args.max_model_len

    config.data.path = args.data_file
    config.data.prompt_key = args.prompt_key
    config.data.image_key = args.image_key
    config.data.decoded_image_key = args.decoded_image_key
    config.data.batch_size = args.batch_size
    config.data.train_batch_size = args.batch_size
    config.data.val_batch_size = args.batch_size
    config.data.max_prompt_length = args.max_prompt_length
    config.data.max_response_length = args.max_new_tokens
    config.data.truncation = "right"
    config.data.return_raw_chat = True
    config.data.return_full_prompt = True
    config.data.filter_overlong_prompts = False
    config.data.n_samples = 1
    config.data.preprocess_num_workers = args.data_num_workers

    config.env.max_steps = 1
    config.env.rollout.n = 1

    config.algorithm.filter_groups.enable = False

    return config


def run_eval(args: argparse.Namespace) -> int:
    ensure_repo_importable()

    from verl import DataProto
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from alphaapollo.core.generation.multi_turn_rollout import TrajectoryCollector

    if not ray.is_initialized():
        repo_root = ensure_repo_importable()
        local_verl_path = str(repo_root / "alphaapollo" / "core" / "generation")
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        worker_pythonpath = f"{local_verl_path}:{str(repo_root)}"
        if existing_pythonpath:
            worker_pythonpath = f"{worker_pythonpath}:{existing_pythonpath}"
        ray.init(
            num_cpus=args.ray_num_cpus,
            _temp_dir="/gz-data/logs/ray_tmp",
            runtime_env={
                "env_vars": {"PYTHONPATH": worker_pythonpath},
            },
        )

    config = build_config(args)

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    processor = hf_processor(args.model_path, trust_remote_code=True, use_fast=True)

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RLHFDataset(
        data_files=args.data_file,
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
    )

    total_dataset = len(dataset)
    total = min(total_dataset, args.max_samples) if args.max_samples > 0 else total_dataset

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / args.predictions_file
    metrics_file = out_dir / args.metrics_file
    case_file = out_dir / args.case_study_file

    print(f"[info] dataset_len: {total_dataset}")
    print(f"[info] eval_samples: {total}")
    print(f"[info] output_dir: {out_dir}")

    # Add a hard delay before starting evaluation to ensure resources are ready
    # debug尝试: 防止 worker 未处于就绪态导致 assertion error
    import time
    delay_seconds = 10
    print(f"[info] Hard delay: sleeping for {delay_seconds} seconds before worker initialization...")
    time.sleep(delay_seconds)

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    # Avoid oversized CPU bundles from RayResourcePool default max_colocate_count=10.
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        max_colocate_count=args.max_colocate_count,
    )
    worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda")
    worker_group.init_model()

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    envs = OneStepMathVistaEnvManager()

    records: list[dict[str, Any]] = []

    with predictions_file.open("w", encoding="utf-8") as fout:
        for start in tqdm(range(0, total, args.batch_size), desc="Evaluating"):
            stop = min(start + args.batch_size, total)
            sample_indices = list(range(start, stop))

            rows = [dict(dataset.dataframe[i]) for i in sample_indices]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.data_num_workers) as executor:
                samples = list(executor.map(dataset.__getitem__, sample_indices))

            for row, sample in zip(rows, samples):
                image_items = dataset._pick_image_items(dict(row))
                image_item = image_items[0] if image_items is not None and len(image_items) > 0 else None
                query = row.get(args.prompt_key, "")
                if not isinstance(query, str):
                    query = normalize_text(query)
                sample["env_kwargs"] = {
                    "query": query,
                    "image": image_item,
                }

            # Ensure data_source and raw_prompt exist (required by preprocess_single_sample)
            for row, sample in zip(rows, samples):
                if "data_source" not in sample:
                    sample["data_source"] = "mathvista"
                if "raw_prompt" not in sample:
                    query = row.get(args.prompt_key, "")
                    if not isinstance(query, str):
                        query = normalize_text(query)
                    sample["raw_prompt"] = [{"role": "user", "content": query}]

            batch_dict = collate_fn(samples)
            batch = DataProto.from_single_dict(batch_dict)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = []
            if "raw_prompt_ids" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt_ids")
            if "data_source" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("data_source")
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")

            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": args.do_sample,
                "validate": False,
            }

            gen_batch_output = traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=worker_group,
                envs=envs,
                is_train=False,
            )

            # Keep the latest step per local index; with env.max_steps=1 this is exactly one response.
            local_idx_to_response: dict[int, str] = {}
            for i in range(len(gen_batch_output)):
                item = gen_batch_output[i]
                local_idx = int(item.non_tensor_batch["index"])
                response_ids = item.batch["responses"]
                if isinstance(response_ids, torch.Tensor):
                    response_ids = response_ids.detach().cpu().tolist()
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                local_idx_to_response[local_idx] = response

            for local_idx, (global_idx, row) in enumerate(zip(sample_indices, rows)):
                response = local_idx_to_response.get(local_idx, "")
                query = row.get(args.prompt_key, "")
                if not isinstance(query, str):
                    query = normalize_text(query)

                question_type = normalize_text(row.get("question_type", ""))
                answer_type = normalize_text(row.get("answer_type", ""))
                choices = row.get("choices")
                choices = choices if isinstance(choices, list) else None
                precision = row.get("precision", None)
                gold_answer = row.get("answer", "")

                final_segment = extract_final_segment(response)

                if question_type == "multi_choice":
                    letter = extract_multi_choice_letter(final_segment, len(choices or []))
                    if letter is None:
                        letter = extract_multi_choice_letter(response, len(choices or []))
                    extraction = letter if letter is not None else final_segment
                else:
                    extraction = final_segment

                prediction = normalize_extraction(
                    extraction=extraction,
                    choices=choices,
                    question_type=question_type,
                    answer_type=answer_type,
                    precision=precision,
                )
                correct = safe_equal(prediction, gold_answer)

                rec = {
                    "idx": global_idx,
                    "pid": row.get("pid", str(global_idx)),
                    "question": row.get("question", ""),
                    "query": query,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "choices": choices,
                    "precision": precision,
                    "gold_answer": gold_answer,
                    "response": response,
                    "final_segment": final_segment,
                    "extraction": extraction,
                    "prediction": prediction,
                    "correct": bool(correct),
                    "metadata": row.get("metadata") or {},
                }
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Explicitly release GPU tensors to prevent VRAM accumulation across batches
            del gen_batch_output, gen_batch, batch, batch_dict, samples, rows
            torch.cuda.empty_cache()

            # Force Python GC to drop any lingering references before Ray GC runs
            import gc
            gc.collect()

            # Evict all unreferenced objects from Ray's plasma object store.
            # Ray's GC is reference-counted: once all Python ObjectRef handles are
            # deleted above, the objects become eligible for eviction, but the
            # plasma store won't actually reclaim the pages until we nudge it.
            try:
                # get_all_reference_counts returns {object_id: ref_count}; any
                # entry with ref_count == 0 is already unreferenced and will be
                # freed by the subsequent internal_kv flush.
                unreferenced = [
                    oid for oid, cnt in ray._private.state.state.get_all_reference_counts().items()
                    if cnt == 0
                ]
                if unreferenced:
                    ray._private.internal_api.free(unreferenced, local_only=True)
            except Exception:
                # Ray internal APIs can change across versions; fail silently so
                # the eval loop is never interrupted by cleanup errors.
                pass

    overall_correct = sum(1 for r in records if r.get("correct"))
    overall_total = len(records)
    overall_acc = (overall_correct / overall_total) if overall_total else 0.0

    metrics = {
        "model_path": args.model_path,
        "data_file": args.data_file,
        "num_samples": overall_total,
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_acc,
        },
        "by_question_type": compute_group_accuracy(records, "question_type"),
        "by_task": compute_group_accuracy(records, "task"),
        "by_context": compute_group_accuracy(records, "context"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_case_study(records, case_file, num_success=args.case_success, num_fail=args.case_fail)

    print("\n===== Evaluation Summary =====")
    print(f"Overall Accuracy: {overall_acc:.4%} ({overall_correct}/{overall_total})")
    print(f"Predictions JSONL: {predictions_file}")
    print(f"Metrics JSON: {metrics_file}")
    print(f"Case Study Log: {case_file}")

    envs.close()
    ray.shutdown()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathVista eval with AlphaApollo rollout_loop + vLLM")
    parser.add_argument("--model-path", default="/gz-data/qwen3vl_2b", help="Local model path")
    parser.add_argument(
        "--data-file",
        default="/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet",
        help="MathVista testmini parquet file",
    )
    parser.add_argument("--prompt-key", default="query", help="Prompt key in parquet")
    parser.add_argument("--image-key", default="image", help="Image key in parquet")
    parser.add_argument("--decoded-image-key", default="decoded_image", help="Decoded image key in parquet")

    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max-prompt-length", type=int, default=4096, help="Max prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generated tokens")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples; <=0 means full set")

    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k for vLLM")

    parser.add_argument("--n-gpus-per-node", type=int, default=1, help="GPUs used by rollout worker group")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--ray-num-cpus", type=int, default=8, help="Ray CPU resources")
    parser.add_argument("--data-num-workers", type=int, default=4, help="ThreadPoolExecutor workers for parallel dataset __getitem__")
    parser.add_argument(
        "--max-colocate-count",
        type=int,
        default=1,
        help="Ray placement-group colocate factor; set 1 to request CPU:1+GPU:1 per worker",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM max_model_len")
    parser.add_argument("--free-cache-engine", action="store_true", help="Enable vLLM cache engine free/rebuild")

    parser.add_argument("--output-dir", default="logs/mathvista_testmini_eval_rollout_vllm", help="Output directory")
    parser.add_argument("--predictions-file", default="predictions.jsonl", help="Prediction log file")
    parser.add_argument("--metrics-file", default="metrics.json", help="Metrics file")
    parser.add_argument("--case-study-file", default="case_study.md", help="Case study file")
    parser.add_argument("--case-success", type=int, default=5, help="Number of success cases in case study")
    parser.add_argument("--case-fail", type=int, default=5, help="Number of fail cases in case study")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_eval(args)
    except Exception as exc:
        import traceback
        print("[error] evaluation failed:", repr(exc))
        traceback.print_exc()
        try:
            ray.shutdown()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
