# -*- coding: utf-8 -*-
"""
run_experiment.py —— 实验命令行入口

用法：
    python scripts/run_experiment.py configs/exp_seedH_glm.yaml

这个脚本只做两件事：
1. 解析命令行参数（配置文件路径）；
2. 调用 src.runners.eval_runner.run_eval(config_path)。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    # 确保可以通过 `src.` 导入你的代码（简单粗暴版）
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.runners.eval_runner import run_eval  # 延迟导入，避免循环

    parser = argparse.ArgumentParser(
        description="Run one conformity experiment (multi-model, multi-protocol, 3+1)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to yaml config file, e.g. configs/exp_seedH_glm.yaml",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        # 允许用户从项目根目录/其他目录调用时使用相对路径
        # 尝试在 project_root 下再找一遍
        candidate = project_root / config_path
        if candidate.exists():
            config_path = str(candidate)
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    run_eval(config_path)


if __name__ == "__main__":
    main()
