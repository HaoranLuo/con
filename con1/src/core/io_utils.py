# -*- coding: utf-8 -*-
"""
io_utils.py —— 通用文件读写 & 日志工具

功能：
- read_jsonl / write_jsonl：读写 jsonlines 文件
- write_csv：写 csv 文件
- save_xlsx：写 xlsx 文件（需要安装 pandas + openpyxl）
- ensure_dir：保证目录存在
- setup_logger：统一日志配置

注意：
- 这里不依赖任何模型 / 协议逻辑，纯 IO 工具；
- 可在 core / runners 中到处复用。
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# pandas 是可选依赖：有则用，无则在 save_xlsx 时报友好错误
try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None
    _HAS_PANDAS = False


# ===================== 路径 & 目录 =====================

def ensure_dir(path: Union[str, Path]) -> None:
    """
    保证给定路径所在目录存在：
    - 如果 path 是文件路径，则创建其父目录；
    - 如果 path 本身就是目录，则创建该目录；
    """
    p = Path(path)
    if p.suffix:  # 有后缀，认为是文件
        dir_path = p.parent
    else:
        dir_path = p
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


# ===================== JSONL 读写 =====================

def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    读取 jsonlines 文件（每行一个 JSON 对象），返回 dict 列表。
    """
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    """
    流式读取 jsonlines（生成器），适合大文件。
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _to_plain_obj(obj: Any) -> Any:
    """
    将 dataclass / 非原生类型 转换成可 JSON 序列化的对象。
    """
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def write_jsonl(
    path: Union[str, Path],
    rows: Iterable[Union[Dict[str, Any], Any]],
    ensure_ascii: bool = False,
) -> None:
    """
    将一组 dict 或 dataclass 写入 jsonlines 文件。
    - ensure_ascii=False 可以保留中文。
    """
    p = Path(path)
    ensure_dir(p)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            row_obj = _to_plain_obj(row)
            f.write(json.dumps(row_obj, ensure_ascii=ensure_ascii))
            f.write("\n")


# ===================== CSV / XLSX 写出 =====================

def write_csv(
    path: Union[str, Path],
    rows: Iterable[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """
    将一组 dict 写出为 csv 文件。
    - fieldnames 若为 None，则使用第一行的 key 排序。
    """
    p = Path(path)
    ensure_dir(p)

    rows = list(rows)
    if not rows:
        # 空文件也写个 header（若给了 fieldnames）
        with p.open("w", encoding="utf-8", newline="") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return

    if fieldnames is None:
        # 用第一行的 key 顺序作为列名
        first = rows[0]
        fieldnames = list(first.keys())

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_xlsx(
    data: Union[List[Dict[str, Any]], "pd.DataFrame"],
    path: Union[str, Path],
    sheet_name: str = "Sheet1",
) -> None:
    """
    将数据保存为 xlsx 文件。
    - data 可以是 list[dict] 或 pandas.DataFrame；
    - 需要安装 pandas 和 openpyxl。
    """
    if not _HAS_PANDAS:
        raise ImportError(
            "save_xlsx 需要安装 pandas，请先 `pip install pandas openpyxl`。"
        )

    p = Path(path)
    ensure_dir(p)

    if isinstance(data, list):
        if not data:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(data)
    else:
        df = data  # type: ignore

    df.to_excel(p, index=False, sheet_name=sheet_name)


# ===================== 日志配置 =====================

def setup_logger(
    name: str = "conformity",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    配置一个带控制台输出的 logger，必要时加文件输出。
    - name:     logger 名称；
    - log_file: 若不为 None，则将日志同时写入文件；
    - level:    日志等级（默认 INFO）。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 避免重复输出到 root

    # 如果已经有 handler，就不重复添加（防止多次 setup）
    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # 文件输出（可选）
        if log_file is not None:
            log_file = Path(log_file)
            ensure_dir(log_file)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_fmt = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

    return logger


# ===================== 小工具 =====================

def timestamp_str() -> str:
    """
    返回当前时间的紧凑字符串形式，适合拼进文件名：
    例如：2025-11-14_23-05-12
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
