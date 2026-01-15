# -*- coding: utf-8 -*-
"""
run_taxonomy_classification.py
位置：src/runners/run_taxonomy_classification.py
功能：使用纯文本模型对 VQA 数据集进行二分类 (Perception vs Reasoning)
"""
import sys
import os

# ================= 关键修复开始 =================
# 获取当前脚本的绝对路径
current_path = os.path.abspath(__file__)
# 获取项目根目录 (向上退 3 层：runners -> src -> project)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
# 将项目根目录加入 python path
sys.path.append(project_root)
# ================= 关键修复结束 =================

import time
import argparse
import importlib
from pathlib import Path
from tqdm import tqdm

# 现在可以正常导入 src 了
from src.core.cfg import load_cfg
from src.core.io_utils import read_jsonl, write_jsonl, setup_logger
from src.adapters.base import VLModel

# === 二分类 Prompt ===
TAXONOMY_PROMPT = (
    "Analyze the provided question intended for a Visual Question Answering task.\n"
    "Classify the question into exactly one of the following two categories based on the cognitive skills required:\n\n"
    "1. **Perception**: The answer can be found DIRECTLY by looking at the image. Requires simple recognition, counting, color identification, or OCR. No complex thinking needed.\n"
    "2. **Reasoning**: The answer requires multi-step logic, calculation, spatial relationship inference, external knowledge (common sense/facts), or synthesizing visual clues to reach a conclusion.\n\n"
    "Output ONLY one word: 'Perception' or 'Reasoning'."
)


def load_model(model_cfg) -> VLModel:
    module_name = f"src.adapters.{model_cfg.adapter}"
    module = importlib.import_module(module_name)
    return getattr(module, "create_model")(name=model_cfg.name, **model_cfg.params)


def run_classify(config_path: str):
    # 处理配置文件的路径（支持相对路径和绝对路径）
    if not os.path.isabs(config_path):
        # 如果是相对路径，默认它是相对于项目根目录的
        config_path = os.path.join(project_root, config_path)

    cfg = load_cfg(config_path)
    logger = setup_logger(name="taxonomy_binary")
    logger.info(f"Loaded config: {config_path}")

    # 1. 加载模型
    model_cfg = cfg.models[0]
    logger.info(f"Loading model: {model_cfg.name} (ID: {model_cfg.params.get('model')})")
    model = load_model(model_cfg)

    # 2. 读取数据 (处理相对于 project_root 的路径)
    data_path = cfg.data.path
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    rows = read_jsonl(data_path)
    text_key = cfg.data.text_key
    logger.info(f"Processing {len(rows)} samples from {data_path}")

    results = []

    # 3. 遍历分类
    for row in tqdm(rows, desc="Classifying (P vs R)"):
        question = row.get(text_key, "")
        if not question:
            question = row.get("question", "")

        full_input = f"{TAXONOMY_PROMPT}\n\nQuestion: {question}"
        task_type = "Reasoning"

        for _ in range(3):
            try:
                output = model.predict(image=None, prompt=full_input)
                out_clean = output.strip().replace(".", "").lower()

                if "perception" in out_clean:
                    task_type = "Perception"
                elif "reasoning" in out_clean:
                    task_type = "Reasoning"
                break
            except Exception as e:
                time.sleep(1)

        new_row = row.copy()
        new_row["task_type"] = task_type
        results.append(new_row)

    # 4. 保存结果
    p = Path(data_path)
    output_path = p.parent / f"classified_{p.name}"

    write_jsonl(output_path, results)

    # 统计
    counts = {"Perception": 0, "Reasoning": 0}
    for r in results:
        t = r.get("task_type", "Reasoning")
        counts[t] = counts.get(t, 0) + 1

    logger.info(f"Saved to: {output_path}")
    logger.info(f"Stats: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to yaml config file (relative to project root).")
    args = parser.parse_args()
    run_classify(args.config)