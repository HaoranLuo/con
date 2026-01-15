# -*- coding: utf-8 -*-
"""
glm_api.py —— 智谱 GLM 多模态模型适配器（GLM-4V-Flash / GLM-4.1V-Thinking / GLM-4.5V 等）

依赖：
    pip install zhipuai pillow

配置示例（YAML）：
  models:
    - name: "GLM-4V-Flash"
      adapter: "glm_api"
      params:
        api_key: "YOUR_ZHIPU_API_KEY"
        model: "glm-4v-flash"
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import VLModel



def _image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    """PIL.Image -> data:image/...;base64,xxxx"""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/" + fmt.lower()
    return f"data:{mime};base64,{b64}"


class GLMApiModel(VLModel):
    """
    智谱多模态模型（GLM-4V 系列）的 API 适配器。
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        model: str = "glm-4v",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        self.api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZHIPU_API_KEY / ZHIPUAI_API_KEY 未设置，且未在 params.api_key 中提供。")

        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

        # 尝试初始化官方 SDK
        from zhipuai import ZhipuAI  # type: ignore

        self.client = ZhipuAI(api_key=self.api_key)

    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        # 构造多模态 content
        content: List[Dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        if image is not None:
            url = _image_to_data_url(image, fmt="JPEG")
            content.append(
                {"type": "image_url", "image_url": {"url": url}}
            )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        # 按官方返回结构尝试解析；失败就直接转成字符串
        try:
            choice = resp.choices[0]
            msg = choice.message
            # 新版 SDK 常见结构： message.content 为 str 或 list
            content_obj = getattr(msg, "content", None)
            if isinstance(content_obj, str):
                return content_obj.strip()
            if isinstance(content_obj, list):
                texts = []
                for item in content_obj:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(str(item["text"]))
                if texts:
                    return "\n".join(texts).strip()
            return str(resp)
        except Exception:
            return str(resp)


def create_model(name: str, **params: Any) -> VLModel:
    """给 eval_runner 调用的工厂函数"""
    return GLMApiModel(name=name, **params)
