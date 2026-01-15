# -*- coding: utf-8 -*-
"""
cfg.py —— 实验配置（Config）定义与加载

主要功能：
- 定义一组 dataclass：ModelCfg / DataCfg / GroupingCfg / SaveCfg / Cfg
- 从 yaml 文件加载配置并做基本校验
- 统一展开相对路径，便于在 runner 中使用

典型用法：
    from core.cfg import load_cfg

    cfg = load_cfg("configs/exp_seedH_qwen.yaml")
    print(cfg.exp_name, cfg.data.path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "cfg.py 需要 PyYAML 支持，请先安装：`pip install pyyaml`"
    ) from e


# ===================== dataclass 定义 =====================

@dataclass
class DataCfg:
    """数据相关配置。"""
    path: str                     # experiment_ready(3).jsonl 路径
    image_key: str = "image_b64"  # jsonl 中图像字段名
    text_key: str = "question"    # jsonl 中问题字段名
    label_key: str = "answer"     # jsonl 中正确答案字段名（字母或文本，取决于你的预处理）


@dataclass
class GroupingCfg:
    """一图多题 3+1 分组相关配置。"""
    context_k: int = 3                # 群体题目个数（默认 3）
    group_correct_ratio: float = 1.0  # 群体答案正确比例：1.0=全对，0.0=全错，0.5=一半对一半错


@dataclass
class ModelCfg:
    """单个模型的配置。"""
    name: str                     # 模型逻辑名称（如 "GLM-4V-Flash"）
    adapter: str                  # 适配器模块名（如 "glm4v_flash"）
    params: Dict[str, Any] = field(default_factory=dict)  # 传给适配器 __init__ 的参数


@dataclass
class SaveCfg:
    """结果保存相关配置。"""
    dir: str                      # 如 "outputs/results"
    log_dir: Optional[str] = None # 如 "outputs/logs"
    write_xlsx: bool = False      # 是否额外导出 xlsx（数据大时可关掉）


@dataclass
class Cfg:
    """
    总配置对象。对应一份 yaml 配置文件。
    """
    exp_name: str
    seed: int
    data: DataCfg
    grouping: GroupingCfg
    protocols: List[str]
    models: List[ModelCfg]
    save: SaveCfg

    # 可选：额外元信息
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- 便捷属性 ----

    @property
    def save_dir(self) -> Path:
        return Path(self.save.dir)

    @property
    def log_dir(self) -> Optional[Path]:
        return Path(self.save.log_dir) if self.save.log_dir else None


# ===================== 加载与校验 =====================

def _ensure_abs(path: str, base_dir: Path) -> str:
    """
    若 path 为相对路径，则基于 base_dir 拼成绝对路径；
    若本来就是绝对路径，则原样返回。

    注意：在本项目中，我们约定：
      - data.path、save.dir、save.log_dir 的相对路径
        都是相对于“项目根目录”（project/），
        而不是相对于 configs/ 目录。
    """
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


def _parse_data_cfg(raw: Dict[str, Any], base_dir: Path) -> DataCfg:
    if "path" not in raw:
        raise KeyError("配置文件缺少 data.path 字段")
    data = DataCfg(
        path=_ensure_abs(str(raw["path"]), base_dir),
        image_key=str(raw.get("image_key", "image_b64")),
        text_key=str(raw.get("text_key", "question")),
        label_key=str(raw.get("label_key", "answer")),
    )
    return data


def _parse_grouping_cfg(raw: Dict[str, Any]) -> GroupingCfg:
    return GroupingCfg(
        context_k=int(raw.get("context_k", 3)),
        group_correct_ratio=float(raw.get("group_correct_ratio", 1.0)),
    )


def _parse_model_list(raw_list: Any) -> List[ModelCfg]:
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError("配置文件中的 models 字段必须是非空列表")

    models: List[ModelCfg] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise ValueError("models 列表中的每一项必须是字典")
        name = item.get("name")
        adapter = item.get("adapter")
        if not name or not adapter:
            raise ValueError("models 每一项必须包含 name 和 adapter 字段")
        params = item.get("params") or {}
        models.append(ModelCfg(name=str(name), adapter=str(adapter), params=dict(params)))
    return models


def _parse_save_cfg(raw: Dict[str, Any], base_dir: Path) -> SaveCfg:
    if "dir" not in raw:
        raise KeyError("配置文件缺少 save.dir 字段（结果输出目录）")
    save_dir = _ensure_abs(str(raw["dir"]), base_dir)
    log_dir_raw = raw.get("log_dir")
    log_dir: Optional[str]
    if log_dir_raw is None:
        log_dir = None
    else:
        log_dir = _ensure_abs(str(log_dir_raw), base_dir)

    return SaveCfg(
        dir=save_dir,
        log_dir=log_dir,
        write_xlsx=bool(raw.get("write_xlsx", False)),
    )


def load_cfg(path: str | os.PathLike) -> Cfg:
    """
    从 yaml 文件加载配置。
    - 会对关键字段做存在性校验；
    - 会将 data.path / save.dir / save.log_dir 转为绝对路径；
    - 其余未知字段会被放到 Cfg.extra 中（便于以后扩展）。

    路径约定：
    - 假设配置文件位于 project/configs/xxx.yaml；
    - 则 data.path / save.dir / save.log_dir 中的相对路径
      一律视为相对于 project/ 根目录。
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    # 配置文件所在目录，如 project/configs/
    cfg_dir = cfg_path.parent
    # 约定项目根目录为 configs 的上一层：project/
    project_root = cfg_dir.parent

    with cfg_path.open("r", encoding="utf-8") as f:
        raw_all: Dict[str, Any] = yaml.safe_load(f) or {}

    # 必要字段
    if "exp_name" not in raw_all:
        raise KeyError("配置文件缺少 exp_name 字段")
    if "seed" not in raw_all:
        raise KeyError("配置文件缺少 seed 字段")
    if "data" not in raw_all:
        raise KeyError("配置文件缺少 data 配置块")
    if "grouping" not in raw_all:
        # 可以给个默认的空 dict
        raw_all["grouping"] = {}
    if "protocols" not in raw_all:
        raise KeyError("配置文件缺少 protocols 字段")
    if "models" not in raw_all:
        raise KeyError("配置文件缺少 models 字段")
    if "save" not in raw_all:
        raise KeyError("配置文件缺少 save 配置块")

    exp_name = str(raw_all["exp_name"])
    seed = int(raw_all["seed"])

    # 注意：这里传入的是 project_root，而不是 cfg_dir
    data_cfg = _parse_data_cfg(raw_all["data"], project_root)
    grouping_cfg = _parse_grouping_cfg(raw_all.get("grouping", {}))

    # 协议列表（统一成小写）
    protocols_raw = raw_all["protocols"]
    if not isinstance(protocols_raw, list) or not protocols_raw:
        raise ValueError("protocols 必须是非空列表，例如 ['raw','correct_guidance',...]")
    protocols = [str(p).lower() for p in protocols_raw]

    models_cfg = _parse_model_list(raw_all["models"])
    save_cfg = _parse_save_cfg(raw_all["save"], project_root)

    # 把用掉的 key 移除，其余当 extra
    used_keys = {"exp_name", "seed", "data", "grouping", "protocols", "models", "save"}
    extra = {k: v for k, v in raw_all.items() if k not in used_keys}

    cfg = Cfg(
        exp_name=exp_name,
        seed=seed,
        data=data_cfg,
        grouping=grouping_cfg,
        protocols=protocols,
        models=models_cfg,
        save=save_cfg,
        extra=extra,
    )
    return cfg
