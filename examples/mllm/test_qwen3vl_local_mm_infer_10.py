#!/usr/bin/env python3
"""Run local multimodal inference on first N rows from a parquet file.

Usage:
  cd /gz-data/AlphaApollo-Visual
  python examples/mllm/test_qwen3vl_local_mm_infer_10.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
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
    last_err = None
    model_classes = [AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM]

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


def run_inference(
    model_path: str,
    train_file: str,
    num_samples: int,
    prompt_key: str,
    image_key: str,
    decoded_image_key: str,
    max_prompt_length: int,
    max_new_tokens: int,
    question_preview_chars: int,
) -> int:
    ensure_repo_importable()
    from alphaapollo.core.generation.verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.dataset.vision_utils import process_image

    print("[info] model_path:", model_path)
    print("[info] train_file:", train_file)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path)
    model.eval()

    cfg = OmegaConf.create(
        {
            "prompt_key": prompt_key,
            "image_key": image_key,
            "decoded_image_key": decoded_image_key,
            "max_prompt_length": max_prompt_length,
            "truncation": "right",
            "filter_overlong_prompts": False,
            "return_full_prompt": True,
            "return_raw_chat": True,
        }
    )

    dataset = RLHFDataset(
        data_files=train_file,
        tokenizer=tokenizer,
        processor=processor,
        config=cfg,
    )

    total = min(num_samples, len(dataset))
    print("[info] dataset_len:", len(dataset))
    print("[info] run_samples:", total)

    ok = 0
    fail = 0
    device = infer_model_device(model)

    for i in range(total):
        try:
            raw_row = dataset.dataframe[i]
            image_items = dataset._pick_image_items(dict(raw_row))
            images = [process_image(image_item) for image_item in image_items] if image_items is not None else None

            prompt_value = raw_row.get(prompt_key, "")
            if isinstance(prompt_value, str):
                prompt_text_only = prompt_value
            elif isinstance(prompt_value, dict):
                prompt_text_only = str(prompt_value.get("content", ""))
            elif isinstance(prompt_value, list):
                prompt_text_only = str(prompt_value)
            else:
                prompt_text_only = str(prompt_value)

            content = []
            if images is not None and len(images) > 0:
                content.extend({"type": "image"} for _ in images)
            content.append({"type": "text", "text": prompt_text_only})
            messages = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            model_inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
            model_inputs = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in model_inputs.items()
            }

            with torch.no_grad():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            prompt_len = model_inputs["input_ids"].shape[-1]
            gen_ids = output_ids[:, prompt_len:]
            pred = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            pid = raw_row.get("pid", i)
            answer = raw_row.get("answer", "")
            question = raw_row.get("question", "")
            if question_preview_chars >= 0:
                question_preview = question[:question_preview_chars].replace("\n", " ")
            else:
                question_preview = question.replace("\n", " ")
            image_count = len(images) if images is not None else 0

            print(f"\n===== sample {i + 1}/{total} pid={pid} =====")
            suffix = "..." if question_preview_chars >= 0 and len(question) > question_preview_chars else ""
            print(f"[question] {question_preview}{suffix}")
            print(f"[image_count] {image_count}")
            print(f"[gold] {answer}")
            print(f"[pred] {pred}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"\n[error] sample {i + 1}/{total} failed: {exc!r}")

    print("\n===== summary =====")
    print("[ok]", ok)
    print("[failed]", fail)
    return 0 if fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL local multimodal inference test on first N parquet rows")
    parser.add_argument("--model-path", default="/gz-data/qwen3vl_2b", help="Local model path")
    parser.add_argument(
        "--train-file",
        default="/gz-data/dataset/data/testmini-00000-of-00001-725687bf7a18d64b.parquet",
        help="Parquet file for multimodal inference test",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of rows to run from the beginning")
    parser.add_argument("--prompt-key", default="query", help="Prompt key in parquet")
    parser.add_argument("--image-key", default="image", help="Image path key in parquet")
    parser.add_argument("--decoded-image-key", default="decoded_image", help="Decoded image payload key in parquet")
    parser.add_argument("--max-prompt-length", type=int, default=4096, help="Max prompt length for dataset preprocessing")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens per sample")
    parser.add_argument(
        "--question-preview-chars",
        type=int,
        default=300,
        help="How many chars of question to print; -1 prints full question",
    )
    args = parser.parse_args()

    try:
        return run_inference(
            model_path=args.model_path,
            train_file=args.train_file,
            num_samples=args.num_samples,
            prompt_key=args.prompt_key,
            image_key=args.image_key,
            decoded_image_key=args.decoded_image_key,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            question_preview_chars=args.question_preview_chars,
        )
    except Exception as exc:  # noqa: BLE001
        print("[error] test failed:", repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
