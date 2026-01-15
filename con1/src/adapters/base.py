# src/adapters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class VLModel(ABC):
    """
    统一的视觉语言模型接口：
      - 输入：PIL.Image 或 None + 文本 prompt
      - 输出：模型返回的纯文本字符串
    """

    def __init__(self, name: str, **meta):
        self.name = name
        self.meta = meta

    @abstractmethod
    def predict(self, image: Optional[Image.Image], prompt: str) -> str:
        raise NotImplementedError
