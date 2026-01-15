# -*- coding: utf-8 -*-
"""
claude_api.py —— Claude 3 Opus 多模态适配器

依赖：
    pip install anthropic pillow

YAML 示例：
  models:
    - name: "Claude-3-Opus"
      adapter: "claude_api"
      params:
        api_key: "YOUR_ANTHROPIC_API_KEY"
        model: "claude-3-opus-20240229"
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import VLModel

import anthropic  # type: ignore


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ClaudeApiModel(VLModel):
    """
    Claude 3 系列的多模态适配器（示例以 Opus 为主）。
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY 未设置，且未在 params.api_key 中提供。")

        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        # Claude messages API 的 content 是一个 list
        content: List[Dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        if image is not None:
            img_b64 = _image_to_base64(image, fmt="PNG")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                }
            )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        try:
            # resp.content 是一个 block 列表，每个 block 里有 text
            texts: List[str] = []
            for block in resp.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            if texts:
                return "\n".join(texts).strip()
            return str(resp)
        except Exception:
            return str(resp)


def create_model(name: str, **params: Any) -> VLModel:
    return ClaudeApiModel(name=name, **params)
