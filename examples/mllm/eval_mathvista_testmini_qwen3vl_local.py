#!/usr/bin/env python3
"""MathVista testmini full evaluation with AlphaApollo + local Qwen3-VL.

Usage:
  cd /gz-data/AlphaApollo-Visual
  conda run -n base python examples/mllm/eval_mathvista_testmini_qwen3vl_local.py
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import difflib
import json
import re
import sys
from collections import defaultdict
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)


def ensure_repo_importable() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def infer_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def load_model(model_path: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_classes = [AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM]
    last_err = None
    for model_cls in model_classes:
        try:
            model = model_cls.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            print(f"[info] loaded_model_class: {model.__class__.__name__} (via {model_cls.__name__})")
            return model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Failed to load model from {model_path}: {last_err!r}")


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

    # Prioritize explicit final answer markers.
    marker_patterns = [
        r"(?:final\s*answer|answer|答案|最终答案)\s*[:：]\s*(.+)$",
        r"(?:therefore|thus|so)\s*,?\s*the\s*answer\s*is\s*(.+)$",
    ]
    for pat in marker_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fall back to the last sentence-ish chunk.
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
    """
    先用正则精准提取（如 extract_multi_choice_letter），
    如果正则实在抓不到东西，最后才用这个 get_most_similar 作为一个保底方案，
    防止因为一个标点符号或大小写的细微差别导致模型丢分。
    """
    if not choices:
        return ""
    # difflib 库计算相似度
    sims = [difflib.SequenceMatcher(a=normalize_text(candidate).lower(), b=normalize_text(c).lower()).ratio() for c in choices]
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

        # If model outputs full option text directly.
        # 防止输出答案而不是前面的 A B C D
        for c in choices:
            if normalize_text(c).lower() == extraction.lower():
                return normalize_text(c)
        # 模糊度匹配
        return normalize_text(get_most_similar(extraction, choices)) if choices else extraction

    if answer_type == "integer":
        num = find_last_number(extraction)
        if num is None:
            return None
        try:
            return str(int(float(num)))
        except Exception:  # noqa: BLE001
            return None

    if answer_type == "float":
        num = find_last_number(extraction)
        if num is None:
            return None
        try:
            p = int(float(precision)) if precision is not None else 2
            return str(round(float(num), p))
        except Exception:  # noqa: BLE001
            return None

    if answer_type == "list":
        # Prefer literal python-list parsing from bracket span.
        m = re.search(r"\[[^\]]*\]", extraction)
        candidate = m.group(0) if m else extraction
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, list):
                return str(obj)
        except Exception:  # noqa: BLE001
            pass
        # Fallback: gather numbers and format list.
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", extraction)
        if nums:
            return str([float(x) if "." in x else int(x) for x in nums])
        return None

    # text / default
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


def run_eval(args: argparse.Namespace) -> int:
    ensure_repo_importable()

    from alphaapollo.core.generation.verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.dataset.vision_utils import process_image

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / args.predictions_file
    metrics_file = out_dir / args.metrics_file
    case_file = out_dir / args.case_study_file

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_model(args.model_path)
    model.eval()
    device = infer_model_device(model)

    cfg = OmegaConf.create(
        {
            "prompt_key": args.prompt_key,
            "image_key": args.image_key,
            "decoded_image_key": args.decoded_image_key,
            "max_prompt_length": args.max_prompt_length,
            "truncation": "right",
            "filter_overlong_prompts": False,
            "return_full_prompt": True,
            "return_raw_chat": True,
        }
    )

    dataset = RLHFDataset(
        data_files=args.data_file,
        tokenizer=tokenizer,
        processor=processor,
        config=cfg,
    )

    dataset_total = len(dataset)
    total = min(dataset_total, args.max_samples) if args.max_samples > 0 else dataset_total
    print(f"[info] dataset_len: {dataset_total}")
    print(f"[info] eval_samples: {total}")
    print(f"[info] output_dir: {out_dir}")
    print(f"[info] prefetch_workers: {args.prefetch_workers}")
    print(f"[info] prefetch_size: {args.prefetch_size}")

    records: list[dict[str, Any]] = []

    def _prepare_sample(sample_idx: int) -> dict[str, Any]:
        raw_row = dataset.dataframe[sample_idx]
        row = dict(raw_row)

        image_items = dataset._pick_image_items(dict(raw_row))
        images = [process_image(image_item) for image_item in image_items] if image_items is not None else None

        query = row.get(args.prompt_key, "")
        if not isinstance(query, str):
            query = normalize_text(query)

        return {
            "idx": sample_idx,
            "row": row,
            "query": query,
            "images": images,
        }

    executor: concurrent.futures.ThreadPoolExecutor | None = None
    if args.prefetch_workers > 0:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch_workers)

    pending: deque[Any] = deque()
    index_iter = iter(range(total))

    def _enqueue_prefetch() -> None:
        while len(pending) < args.prefetch_size:
            try:
                idx = next(index_iter)
            except StopIteration:
                break
            if executor is not None:
                pending.append(executor.submit(_prepare_sample, idx))
            else:
                pending.append(_prepare_sample(idx))

    _enqueue_prefetch()

    with predictions_file.open("w", encoding="utf-8") as fout:
        for _ in tqdm(range(total), desc="Evaluating"):
            item_or_future = pending.popleft()
            if executor is not None:
                item = item_or_future.result()
            else:
                item = item_or_future
            _enqueue_prefetch()

            i = item["idx"]
            row = item["row"]
            query = item["query"]
            images = item["images"]

            content = []
            if images is not None and len(images) > 0:
                content.extend({"type": "image"} for _ in images)
            content.append({"type": "text", "text": query})
            messages = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            model_inputs = processor(text=[prompt_text], images=images, return_tensors="pt")

            moved_inputs: dict[str, Any] = {}
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    if args.pin_memory and v.device.type == "cpu":
                        v = v.pin_memory()
                    moved_inputs[k] = v.to(device, non_blocking=args.non_blocking_to_device)
                elif hasattr(v, "to"):
                    moved_inputs[k] = v.to(device)
                else:
                    moved_inputs[k] = v
            model_inputs = moved_inputs

            # 把不同参数解耦
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
            }
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.inference_mode():
                output_ids = model.generate(
                    **model_inputs,
                    **gen_kwargs,
                )

            prompt_len = model_inputs["input_ids"].shape[-1]
            gen_ids = output_ids[:, prompt_len:]
            response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

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
                "idx": i,
                "pid": row.get("pid", str(i)),
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

    if executor is not None:
        executor.shutdown(wait=True)

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

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MathVista testmini full evaluation with AlphaApollo + Qwen3-VL")
    parser.add_argument("--model-path", default="/gz-data/qwen3vl_2b", help="Local model path")
    parser.add_argument(
        "--data-file",
        default="/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet",
        help="MathVista testmini parquet file",
    )
    parser.add_argument("--prompt-key", default="query", help="Prompt key in parquet")
    parser.add_argument("--image-key", default="image", help="Image path key in parquet")
    parser.add_argument("--decoded-image-key", default="decoded_image", help="Decoded image payload key in parquet")
    parser.add_argument("--max-prompt-length", type=int, default=4096, help="Max prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens per sample")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max number of samples to evaluate; -1 means full set")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument("--prefetch-workers", type=int, default=4, help="Thread workers for CPU sample preprocessing")
    parser.add_argument("--prefetch-size", type=int, default=16, help="Number of prefetched samples queued ahead")
    parser.add_argument("--pin-memory", action="store_true", help="Pin CPU tensors before device transfer")
    parser.add_argument(
        "--non-blocking-to-device",
        action="store_true",
        help="Use non-blocking CPU->GPU copies (best with --pin-memory)",
    )
    parser.add_argument("--output-dir", default="logs/mathvista_testmini_eval", help="Output directory")
    parser.add_argument("--predictions-file", default="predictions.jsonl", help="Per-sample prediction log file")
    parser.add_argument("--metrics-file", default="metrics.json", help="Metrics summary file")
    parser.add_argument("--case-study-file", default="case_study.md", help="Case study markdown file")
    parser.add_argument("--case-success", type=int, default=2, help="Number of success cases")
    parser.add_argument("--case-fail", type=int, default=2, help="Number of failure cases")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_eval(args)
    except Exception as exc:  # noqa: BLE001
        print("[error] evaluation failed:", repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
