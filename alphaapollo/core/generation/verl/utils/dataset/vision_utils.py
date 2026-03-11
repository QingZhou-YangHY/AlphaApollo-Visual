# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def _decode_base64_to_bytes(payload: str) -> bytes:
    # Support both raw base64 payload and RFC2397 data URI payload.
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _to_image_field(value):
    if value is None:
        return None
    if isinstance(value, BytesIO):
        return value
    if isinstance(value, memoryview):
        return BytesIO(value.tobytes())
    if isinstance(value, (bytes, bytearray)):
        return BytesIO(bytes(value))
    if isinstance(value, str):
        # 这里因为 .parquet 文件里可能会直接存 base64 字符串来表示图片
        # Allow inline images provided as data URI.
        if value.startswith("data:"):
            return BytesIO(_decode_base64_to_bytes(value))
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def process_image(image: Union[dict, Image.Image, str, Path]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (bytes, bytearray, memoryview, BytesIO)):
        image = {"image": _to_image_field(image)}

    if isinstance(image, (str, Path)):
        image = {"image": str(image)}

    if not isinstance(image, dict):
        raise TypeError(f"Unsupported image payload type: {type(image)}")

    image = dict(image)

    # Accept parquet-style nested payload: {'decoded_image': {'bytes': ..., 'path': ...}}.
    if "decoded_image" in image and "image" not in image:
        decoded = image["decoded_image"]
        if isinstance(decoded, dict):
            image.update(decoded)

    # Accept explicit base64 fields and normalize into `image` for downstream loader.
    if "base64" in image and "image" not in image:
        image["image"] = _to_image_field(BytesIO(_decode_base64_to_bytes(image["base64"])))
    elif "b64" in image and "image" not in image:
        image["image"] = _to_image_field(BytesIO(_decode_base64_to_bytes(image["b64"])))

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = _to_image_field(image["bytes"])

    if "path" in image and "image" not in image:
        image["image"] = _to_image_field(image["path"])

    if "image" in image:
        image["image"] = _to_image_field(image["image"])

    return fetch_image(image)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(video)
