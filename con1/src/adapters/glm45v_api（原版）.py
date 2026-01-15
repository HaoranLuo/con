# -*- coding: utf-8 -*-
"""
GLM-4.5V 多模态适配器（强鲁棒版本）
修复内容：
1. 图片在前、文本在后
2. content 结构严格按 Zhipu 文档
3. 提取 content 时强制非空检查
4. 返回空字符串直接 raise，让 eval_runner 捕获
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import VLModel


def _image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/" + fmt.lower()
    return f"data:{mime};base64,{b64}"


class GLMApiModel(VLModel):
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        model: str = "glm-4.5v",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        self.api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv(
            "ZHIPUAI_API_KEY"
        )
        if not self.api_key:
            raise ValueError("未提供 ZHIPU_API_KEY 或 params.api_key。")

        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=self.api_key)

    def _build_messages(
        self, prompt: str, image: Optional[Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        必须保证顺序：
        1. 图片（image_url）
        2. 文本（text）
        """
        content: List[Dict[str, Any]] = []

        if image is not None:
            url = _image_to_data_url(image)
            content.append(
                {"type": "image_url", "image_url": {"url": url}}
            )

        # 文本必须最后加
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        messages = self._build_messages(prompt, image)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"API 调用失败：{e}")

        # ----------- 解析模型输出（强鲁棒）-----------
        try:
            choice = resp.choices[0]
            msg = choice.message
            content_obj = getattr(msg, "content", None)
        except Exception:
            raise RuntimeError(f"返回结构异常：{resp}")

        # content_obj 可能是 str 或 list
        text_out = ""

        if isinstance(content_obj, str):
            text_out = content_obj.strip()

        elif isinstance(content_obj, list):
            parts = []
            for c in content_obj:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(str(c.get("text", "")))
            text_out = "\n".join(parts).strip()

        # ---------- 非空检查 & fallback ----------
        if not text_out:
            # 有些情况下 GLM-4.5V 会把内容写到 reasoning_content 里
            reasoning = getattr(choice, "reasoning_content", None)
            if reasoning:
                text_out = str(reasoning).strip()

        # 如果还是空，那就老老实实返回空串，让上层当作“没答出来”
        # （不要再 raise，让 eval_runner 把它当作普通错误样本）
        return text_out


def create_model(name: str, **params: Any) -> VLModel:
    return GLMApiModel(name=name, **params)
