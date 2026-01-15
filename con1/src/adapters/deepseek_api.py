# -*- coding: utf-8 -*-
"""
deepseek_api.py —— DeepSeek-VL2 多模态适配器 (OpenAI 兼容协议)

适用场景：
1. 第三方中转 API（如 Lemon API, SiliconFlow 等）托管的 deepseek-vl2
2. 本地 vLLM 部署的 openai-compatible 接口

修复与增强：
1. 强制支持 base_url 参数
2. 图片在前，文本在后（符合 VL2 最佳实践）
3. 增加非空检查和错误处理
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image
from openai import OpenAI  # type: ignore

from .base import VLModel


def _image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    """PIL.Image -> data:image/...;base64,xxxx"""
    buf = io.BytesIO()
    # 统一转为 JPEG 兼容性更好，且体积稍小
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/" + fmt.lower()
    return f"data:{mime};base64,{b64}"


class DeepSeekVL2Model(VLModel):
    def __init__(
            self,
            name: str,
            api_key: Optional[str] = None,
            model: str = "deepseek-vl2",
            base_url: Optional[str] = None,
            temperature: float = 0.0,
            top_p: float = 0.9,
            max_tokens: int = 1024,
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        # 1. API Key 处理
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未提供 API Key。请在 yaml params.api_key 中配置。")

        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

        # 2. Base URL 处理 (优先使用 yaml 配置，其次默认，但官方通常不支持 VL2)
        # 如果是中转站，务必在 yaml 里填 base_url
        self.base_url = base_url or "https://api.deepseek.com/v1"

        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        content: List[Dict[str, Any]] = []

        # 3. 构造消息：图片在前，文本在后 (VL 模型通用最佳实践)
        if image is not None:
            url = _image_to_data_url(image, fmt="JPEG")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

        if prompt:
            content.append({"type": "text", "text": prompt})

        try:
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

            # 4. 提取结果
            answer = resp.choices[0].message.content
            if not answer:
                return ""
            return answer.strip()

        except Exception as e:
            # 抛出异常让 eval_runner 的重试机制捕获
            raise RuntimeError(f"DeepSeek API 调用失败: {e}")


def create_model(name: str, **params: Any) -> VLModel:
    return DeepSeekVL2Model(name=name, **params)