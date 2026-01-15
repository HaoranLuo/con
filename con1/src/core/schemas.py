# -*- coding: utf-8 -*-
"""
schemas.py —— 数据结构与规范化工具

设计目标：
- 把“原始样本、一图多题 3+1 任务、模型预测结果”用 dataclass 统一起来；
- 提供少量通用小工具（选项字母 ↔ 索引、base64 图像解码）；
- 方便在 runners / metrics / grouping 中复用，不再到处传裸 dict。
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field, asdict
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image


# ===================== 公共常量与工具 =====================

LETTERS: str = ascii_uppercase  # "A"~"Z"


def letter_to_index(letter: str) -> int:
    """
    将选项字母转为 0-based 索引；非法返回 -1。
    例：'A' -> 0, 'C' -> 2
    """
    if not letter:
        return -1
    ch = str(letter).strip().upper()
    try:
        return LETTERS.index(ch)
    except ValueError:
        return -1


def index_to_letter(idx: int) -> str:
    """
    将 0-based 索引转为选项字母；非法返回 ""。
    例：0 -> 'A', 2 -> 'C'
    """
    try:
        return LETTERS[idx]
    except Exception:
        return ""


def ensure_meta(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """保证 meta 一定是 dict，避免 None 判空。"""
    return dict(meta or {})


# ===================== 图像相关 =====================

def strip_data_url_prefix(b64: str) -> str:
    """
    若是 data URL（形如 'data:image/png;base64,xxxx'），去掉前缀；
    否则原样返回。
    """
    if not b64:
        return ""
    s = b64.strip()
    if s.lower().startswith("data:") and "," in s:
        return s.split(",", 1)[1]
    return s


def image_from_base64(b64: str) -> Image.Image:
    """
    将 base64 / data URL 字符串解码为 PIL.Image.Image。
    - 失败时抛出异常，由上游捕获；
    - 不做任何 resize / normalize。
    """
    core = strip_data_url_prefix(b64)
    if not core:
        raise ValueError("Empty base64 string.")
    raw = base64.b64decode(core)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ===================== 原始样本 Schema =====================

@dataclass
class Sample:
    """
    原始单题样本（对应处理后的 experiment_ready(3).jsonl 中的一行）。

    约定字段尽可能通用，不强绑具体数据集：
    - uid:      样本唯一 ID（如 "<image_id>#<q_idx>"），方便后续追踪；
    - image_id: 与图像相关的逻辑 ID（文件名、不带后缀 ID 等）；
    - image_b64: 图像 base64 / data URL（无图任务可为 None）；
    - question: 题干文本；
    - choices:  选项文本列表（索引 0 对应 "A"）；
    - correct_letter: 正确选项字母（"A"~"Z"），为空表示无客观正确答案 / 待标注；
    - task_type: 任务类别（如 "counting" / "attribute_query" / "ocr" 等，可选）；
    - meta:     额外信息（原始字段、数据集名、split 等，完全自由扩展）。
    """
    uid: str
    image_id: Optional[str]
    image_b64: Optional[str]
    question: str
    choices: List[str]
    correct_letter: str
    task_type: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    @property
    def correct_index(self) -> int:
        return letter_to_index(self.correct_letter)

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的 dict。"""
        return asdict(self)


# ===================== 一图多题 3+1 任务 Schema =====================

@dataclass
class GroupedMission:
    """
    “同图多题 3+1” 后生成的一个评测单元（对应 eval_runner 里的单条样本）。

    - mission_id:   任务唯一标识（如 "<image_id>#<test_q_idx>#<protocol>"）；
    - protocol:     使用的协议名称（'raw' / 'correct_guidance' / 'wrong_guidance' / 'trust' / 'doubt'）；
    - prompt:       完整提示词（含历史轮 + 当前轮 + 输出格式说明）；
    - image_b64:    当前测试题使用的图像（通常就是 test 题对应的那张图）；
    - y_true:       正确答案索引（0-based），无正确答案时可为 -1；
    - true_letter:  正确答案字母（与 y_true 对应；若 y_true <0 则为空字符串）；
    - group_key:    分组用的 key，一般对应某张图/某一对 (doc, image)；
    - source_uids:  本任务所依赖的原始样本 uid 列表（历史题 + 测试题），方便回溯；
    - meta:         其他信息（如历史轮摘要、是否“群体全错”等）。
    """
    mission_id: str
    protocol: str
    prompt: str
    image_b64: Optional[str]
    y_true: int
    true_letter: str
    group_key: Optional[str] = None
    source_uids: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_sample_dict_for_eval(self) -> Dict[str, Any]:
        """
        转为 eval 阶段使用的“组合样本” dict：
            {"text": prompt, "image_base64": <b64 or None>}
        以兼容之后的 eval_runner / adapter 接口。
        """
        return {
            "text": self.prompt,
            "image_base64": self.image_b64,
        }

    def to_target_dict_for_eval(self) -> Dict[str, Any]:
        """
        转为 eval 阶段使用的 targets 结构：
            {"y_true": <int>}
        """
        return {"y_true": int(self.y_true)}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================== 模型预测结果 Schema =====================

@dataclass
class PredictionRecord:
    """
    模型在某个 GroupedMission 上的一次预测结果（逐样本记录）。

    - mission_id:         对应的 GroupedMission.mission_id；
    - protocol:           协议名称；
    - model_name:         模型名称（如 'glm-4v-flash', 'Qwen3-VL-8B-Instruct' 等）；
    - inputs:             实际送入模型的文本（通常等于 GroupedMission.prompt）；
    - outputs:            模型原始输出字符串；
    - y_true:             正确答案索引（0-based；由 GroupedMission.y_true 复制）；
    - y_pred:             预测答案索引（0-based；解析失败时为 -1）；
    - true_letter:        正确答案字母；
    - pred_letter:        预测答案字母（解析失败时为空字符串）；
    - image_b64:          传给模型的图像（可选，用于调试/可视化；默认不导出到 csv/jsonl）；
    - is_correct:         是否预测正确（y_true >=0 且 y_true == y_pred）；
    - meta:               其他需要追踪的信息（如运行时间、重试次数、temperature 等）。
    """
    mission_id: str
    protocol: str
    model_name: str

    inputs: str
    outputs: str

    y_true: int
    y_pred: int

    true_letter: str
    pred_letter: str

    image_b64: Optional[str] = None
    is_correct: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 若未显式给出 is_correct，则根据 y_true / y_pred 自动判断
        if self.is_correct is None:
            if self.y_true is not None and self.y_true >= 0 and self.y_pred is not None:
                self.is_correct = (self.y_true == self.y_pred)
            else:
                self.is_correct = False
        # 确保 meta 为 dict
        self.meta = ensure_meta(self.meta)

    def to_legacy_row(self) -> Dict[str, Any]:
        """
        转为“单行风格”的字典，方便直接丢给 pandas.DataFrame：
        字段命名尽量向统一的 excel/CSV 输出靠拢。

        注意：此处 **不再导出 image_base64**，以避免结果文件过大。
        若需要调试图像，可使用 to_dict() 或单独在 meta 中保存。
        """
        row = {
            "mission_id": self.mission_id,
            "protocol": self.protocol,
            "model": self.model_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "true_letter": self.true_letter,
            "pred_letter": self.pred_letter,
            "is_correct": bool(self.is_correct),
            # 不再包含 "image_base64": self.image_b64,
        }
        # 把 meta 展开到顶层（便于筛选）
        for k, v in (self.meta or {}).items():
            if k not in row:
                row[k] = v
        return row

    def to_dict(self) -> Dict[str, Any]:
        """完全展开为 dict（保留 meta 嵌套）。"""
        return asdict(self)


# ===================== 批量组装工具（兼容旧 outputs 结构，可选） =====================

def pack_predictions_as_outputs_dict(
    preds: Sequence[PredictionRecord],
) -> Dict[str, List[Any]]:
    """
    将多条 PredictionRecord 打包为 outputs dict 结构：

        {
          'inputs':      [str,...],
          'outputs':     [str,...],
          'y_pred':      [int,...],
          'y_true':      [int,...],
          'pred_letter': [str,...]
        }

    这样可以直接复用一些旧的保存 / 分析逻辑，
    同时上层代码也可以选择使用逐行的 PredictionRecord 进行更灵活的分析。
    """
    outs: Dict[str, List[Any]] = {
        "inputs": [],
        "outputs": [],
        "y_pred": [],
        "y_true": [],
        "pred_letter": [],
    }
    for p in preds:
        outs["inputs"].append(p.inputs)
        outs["outputs"].append(p.outputs)
        outs["y_pred"].append(p.y_pred)
        outs["y_true"].append(p.y_true)
        outs["pred_letter"].append(p.pred_letter)
    return outs
