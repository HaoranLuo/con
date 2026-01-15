# -*- coding: utf-8 -*-
"""
grouping.py —— 一图多题聚合 & 3+1 任务单元构造

主要功能：
1. group_by_image(samples)：按 image_id / meta 中的 group_key / image_rel 等字段聚合样本；
2. build_mission_units(samples, context_k)：将每个 image 下面的多题切成“前K题做历史 + 最后一题做测试”的 3+1 单元；
3. choose_group_letter_for_test(sample, group_correct_ratio, rng)：根据指定“群体正确概率”选群体答案字母。

本文件不关心：
- 具体协议（raw/correct/wrong/trust/doubt）；
- 不直接生成 prompt，只负责把任务单元组织好，
  后续由 runners + core.protocols 来渲染为文本提示。

依赖：
- core.schemas.Sample：原始单题；
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Any, DefaultDict, Dict, List, Optional
import random

from .schemas import Sample, letter_to_index, index_to_letter


# ===================== 任务单元定义 =====================

@dataclass
class MissionUnit:
    """
    一个“3+1”任务单元（尚未绑定协议 / prompt）：

    - group_key:   分组标识（通常是一张图像的 ID 或 group_key）；
    - history:     历史题目列表（长度为 context_k）；
    - test:        测试题目（单个 Sample）。
    """
    group_key: str
    history: List[Sample]
    test: Sample


# ===================== 分组相关 =====================

# 保留一个候选列表，兜底使用
_POSSIBLE_GROUP_KEYS = [
    "group_key",
    "image_rel",
    "data_id",
    "image_path",
    "img_id",
    "image",
    "img_relpath",
    "source_image",
    "image_id",
]


def _get_group_key(sample: Sample) -> str:
    """
    尽量找到一个稳定的“同图分组” key：

    **根据你的数据优先级做了强化：**
      1. meta["group_key"]           （你已经确认按这个一图四题）
      2. meta["image_rel"]           （你也确认按这个一图四题）
      3. 其它 meta 字段（_POSSIBLE_GROUP_KEYS 中）
      4. sample.image_id 属性（如果 schemas.py 里有）
      5. 最后兜底用 sample.uid（几乎不会用到）
    """
    meta: Dict[str, Any] = sample.meta or {}

    # 1) 明确优先 group_key
    gk = meta.get("group_key")
    if gk:
        return str(gk)

    # 2) 其次 image_rel（同一张图片的相对路径）
    img_rel = meta.get("image_rel")
    if img_rel:
        return str(img_rel)

    # 3) 再尝试其它可能的 meta 字段
    for k in _POSSIBLE_GROUP_KEYS:
        if k in meta and meta[k]:
            return str(meta[k])

    # 4) 再退化到 Sample.image_id
    if getattr(sample, "image_id", None):
        return str(sample.image_id)

    # 5) 兜底（保证一定返回一个 key）
    return sample.uid


def group_by_image(samples: List[Sample]) -> Dict[str, List[Sample]]:
    """
    按“同一张图像”把样本聚合起来。

    返回：dict[group_key -> List[Sample]]
    """
    buckets: DefaultDict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        key = _get_group_key(s)
        buckets[key].append(s)
    return dict(buckets)


# ===================== 3+1 任务构造 =====================

def _sort_group(samples: List[Sample]) -> List[Sample]:
    """
    对同一张图像下的样本做一个稳定排序，方便选择“前K + 最后一题”。

    **根据你的预处理逻辑做了增强：**
      - 优先使用 meta["index_in_group"]（你在处理数据集时已经写入了这个字段）；
      - 其次尝试 meta["q_index"] / "q_idx" / "question_index" / "index" / "question_id"；
      - 若都没有，则用 uid 排序，保证稳定。
    """
    def sort_key(s: Sample):
        meta = s.meta or {}

        # 1) 优先使用 index_in_group（你写切片时就控制了顺序）
        if "index_in_group" in meta:
            try:
                return int(meta["index_in_group"])
            except Exception:
                return str(meta["index_in_group"])

        # 2) 其它可能的索引字段
        for k in ["q_index", "q_idx", "question_index", "index", "question_id"]:
            if k in meta:
                try:
                    return int(meta[k])
                except Exception:
                    return str(meta[k])

        # 3) 兜底用 uid
        return s.uid

    return sorted(samples, key=sort_key)


def _split_history_and_test(sorted_items: List[Sample], context_k: int):
    """
    简单策略：前K题作为历史，最后1题作为测试。
    若不足 K+1 道题，则返回 (None, None)。
    """
    if len(sorted_items) < context_k + 1:
        return None, None
    return sorted_items[:context_k], sorted_items[-1]


def build_mission_units(
    samples: List[Sample],
    context_k: int = 3,
) -> List[MissionUnit]:
    """
    将同图多题样本构造成若干 3+1 任务单元（不含协议和 prompt）。

    参数：
      - samples:    预处理后的 Sample 列表（同一 image_id / group_key 下有多题）；
      - context_k:  历史题目数量，默认 3（即 3+1）。

    返回：
      - mission_units: List[MissionUnit]
    """
    # 1) 按“同一张图 / 同一 group_key”聚合
    buckets = group_by_image(samples)
    mission_units: List[MissionUnit] = []

    for gkey, items in buckets.items():
        # 2) 对同一图像下的题目做一个稳定排序
        sorted_items = _sort_group(items)
        # 3) 切分 “前K” 和 “最后1个”
        hist, test = _split_history_and_test(sorted_items, context_k)
        if not hist or not test:
            continue
        mission_units.append(
            MissionUnit(group_key=gkey, history=list(hist), test=test)
        )

    return mission_units


# ===================== 群体答案选择工具 =====================

def choose_group_letter_for_test(
    test_sample: Sample,
    group_correct_ratio: float = 1.0,
    rng: Optional[random.Random] = None,
) -> str:
    """
    根据给定“群体正确概率”选出群体答案字母，用于各种带指导/从众的协议构造。

    - test_sample:          当前测试题目（必须已经有 choices + correct_letter）；
    - group_correct_ratio:  群体选择正确答案的概率（0.0~1.0）；
                             例：
                               1.0 -> 群体总是给出正确答案
                               0.0 -> 群体总是给出错误答案（从其他选项中随机挑一个）
                               0.5 -> 群体一半概率给对，一半给错
    - rng:                  可选的 random.Random 实例；不传则使用全局 random。

    返回：
      - 群体给出的选项字母（"A"~"Z"），若无法确定则尽量返回一个合法字母。
    """
    rng = rng or random

    n = test_sample.n_choices
    if n <= 0:
        return "A"

    # 正确答案索引 & 字母
    correct_idx = test_sample.correct_index
    if correct_idx < 0 or correct_idx >= n:
        # 如果没有合法的 correct_letter，就随机选一个
        return ascii_uppercase[rng.randrange(n)]

    correct_letter = index_to_letter(correct_idx)

    # 决定这一次群体是否“给对”
    p = max(0.0, min(1.0, float(group_correct_ratio)))
    give_correct = rng.random() < p

    if give_correct or n == 1:
        return correct_letter

    # 从剩余选项中随机挑一个错误答案
    all_letters = [ascii_uppercase[i] for i in range(n)]
    wrong_letters = [L for L in all_letters if L != correct_letter]
    if not wrong_letters:
        return correct_letter
    return rng.choice(wrong_letters)
