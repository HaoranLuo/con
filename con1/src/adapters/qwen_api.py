# -*- coding: utf-8 -*-
"""
Qwen2.5-VL (OpenAI 协议兼容) 多模态适配器
适用：Qwen2.5-VL, GPT-4o, Claude-3.5 (通过中转) 等兼容 OpenAI 格式的模型
特点：
1. 使用标准 openai 库调用
2. 图片/文本结构符合 OpenAI Vision 标准
3. 强鲁棒性错误捕获
4. 支持自定义 base_url (Lemon API 等中转必填)
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from PIL import Image
# 需要安装: pip install openai
from openai import OpenAI, APIConnectionError, APIError

from .base import VLModel


def _image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    """将 PIL Image 转为 data:image/jpeg;base64,xxx 格式"""
    buf = io.BytesIO()
    #以此保证图片不会太大导致超过 token 限制，也可在此处加入 resize 逻辑
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/" + fmt.lower()
    return f"data:{mime};base64,{b64}"


class QwenApiModel(VLModel):
    def __init__(
            self,
            name: str,
            api_key: Optional[str] = None,
            model: str = "Qwen2.5-VL-72B-Instruct",
            temperature: float = 0.0,
            top_p: float = 0.9,
            max_tokens: int = 1024,
            base_url: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        # 1. 获取 API Key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未提供 api_key，请在 yaml 或环境变量中设置。")

        # 2. 获取 Base URL (对于中转 API 非常重要)
        self.base_url = base_url or kwargs.get("base_url")
        if not self.base_url:
             # 如果没有提供 base_url，默认可能是官方 OpenAI，但在 Qwen 语境下最好报错提示
             # 或者你可以设置为阿里云的官方 endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1"
             pass

        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

        # 3. 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            # 可以添加 timeout 等其他参数
            # timeout=60.0,
        )

    def _build_messages(
            self, prompt: str, image: Optional[Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        构建符合 OpenAI Vision 格式的消息体
        """
        user_content: List[Dict[str, Any]] = []

        # 1. 处理图片 (OpenAI 格式推荐图片在前或后均可，通常图片在前较好)
        if image is not None:
            data_url = _image_to_data_url(image)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    # "detail": "high" # Qwen 通常不需要手动指定 detail，自动即可
                }
            })

        # 2. 处理文本
        user_content.append({
            "type": "text",
            "text": prompt
        })

        return [{"role": "user", "content": user_content}]

    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        messages = self._build_messages(prompt, image)

        try:
            # 发起请求
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=False
            )
        except APIConnectionError as e:
            raise RuntimeError(f"网络连接错误 (BaseURL: {self.base_url}): {e}")
        except APIError as e:
            raise RuntimeError(f"API 返回错误: {e}")
        except Exception as e:
            raise RuntimeError(f"未知异常: {e}")

        # ----------- 解析模型输出 -----------
        try:
            choice = resp.choices[0]
            content = choice.message.content
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"解析响应结构失败: {resp} \nError: {e}")

        # 非空处理
        if not content:
            # 某些模型可能返回 null content 但有 refusal
            if hasattr(choice.message, 'refusal') and choice.message.refusal:
                return f"[Refusal]: {choice.message.refusal}"
            return ""

        return content.strip()


def create_model(name: str, **params: Any) -> VLModel:
    return QwenApiModel(name=name, **params)