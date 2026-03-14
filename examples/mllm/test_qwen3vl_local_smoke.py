#!/usr/bin/env python3
"""Minimal local smoke test for Qwen3-VL + RLHFDataset pipeline.

Usage:
  PYTHONPATH=/root/autodl-tmp/AlphaApollo-Visual \
  conda run -n base python examples/rl/test_qwen3vl_local_smoke.py
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoTokenizer


def ensure_repo_importable() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def run_smoke(model_path: str, train_file: str, prompt_key: str, image_key: str) -> int:
    ensure_repo_importable()

    from alphaapollo.core.generation.verl.utils.dataset.rl_dataset import RLHFDataset

    print("[info] model_path:", model_path)
    print("[info] train_file:", train_file)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    try:
        call_params = inspect.signature(processor.__call__).parameters
        mm_support = hasattr(processor, "image_processor") or ("images" in call_params)
    except (TypeError, ValueError):
        mm_support = hasattr(processor, "image_processor")

    print("[info] processor_class:", processor.__class__.__name__)
    print("[info] processor_multimodal_support:", mm_support)

    cfg = OmegaConf.create(
        {
            "prompt_key": prompt_key,
            "image_key": image_key,
            "max_prompt_length": 4096,
            "truncation": "right",
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
        }
    )

    dataset = RLHFDataset(
        data_files=train_file,
        tokenizer=tokenizer,
        processor=processor,
        config=cfg,
    )

    print("[ok] dataset_len:", len(dataset))

    sample = dataset[0]
    print("[ok] sample_keys:", sorted(sample.keys()))

    mm = sample.get("multi_modal_data")
    print("[ok] has_multi_modal_data:", mm is not None)
    if mm is not None:
        print("[ok] multi_modal_keys:", list(mm.keys()))
        image_count = len(mm.get("image", [])) if mm.get("image") is not None else 0
        print("[ok] image_count:", image_count)

    input_ids = sample.get("input_ids")
    attention_mask = sample.get("attention_mask")
    if input_ids is not None and attention_mask is not None:
        print("[ok] input_ids_len:", int(input_ids.shape[-1]))
        print("[ok] attention_mask_len:", int(attention_mask.shape[-1]))

    print("[done] smoke test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL local smoke test")
    parser.add_argument(
        "--model-path",
        default="/gz-data/qwen3vl_2b",
        help="Local model path",
    )
    parser.add_argument(
        "--train-file",
        default="/gz-data/dataset/data/test-00000-of-00002-6b81bd7f7e2065e6.parquet",
        help="Parquet file for RLHFDataset smoke test",
    )
    parser.add_argument("--prompt-key", default="query", help="Prompt key in parquet")
    parser.add_argument("--image-key", default="image", help="Image key in parquet")
    args = parser.parse_args()

    try:
        return run_smoke(
            model_path=args.model_path,
            train_file=args.train_file,
            prompt_key=args.prompt_key,
            image_key=args.image_key,
        )
    except Exception as exc:
        print("[error] smoke test failed:", repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
