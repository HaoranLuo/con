# -*- coding: utf-8 -*-
"""
openai_api.py —— OpenAI 多模态模型适配器（GPT-4o / GPT-5 等）

依赖：
    pip install openai pillow

YAML 示例：
  models:
    - name: "GPT-4o"
      adapter: "openai_api"
      params:
        api_key: "YOUR_OPENAI_API_KEY"
        model: "gpt-4o"
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import VLModel

from openai import OpenAI  # type: ignore


def _image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/" + fmt.lower()
    return f"data:{mime};base64,{b64}"


class OpenAIModel(VLModel):
    """
    OpenAI 多模态模型适配器（GPT-4o / GPT-5 等）。
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 未设置，且未在 params.api_key 中提供。")

        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        content: List[Dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        if image is not None:
            url = _image_to_data_url(image, fmt="PNG")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            return str(resp)


def create_model(name: str, **params: Any) -> VLModel:
    return OpenAIModel(name=name, **params)
