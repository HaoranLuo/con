# -*- coding: utf-8 -*-
"""
eval_runner.py —— 单次实验主流程（多模型 × 多协议 × 一图多题 3+1）

功能概览：
1. load_cfg(config_path) 读取 yaml 配置；
2. 读取 experiment_ready(3).jsonl -> Sample 列表；
3. core.grouping.build_mission_units 做一图多题 3+1；
4. 针对每个 MissionUnit × 协议 构建 GroupedMission + Prompt；
5. 调用模型适配器（adapters.*）进行预测；
6. 将结果写入 jsonl，并打印每个协议的 Accuracy。

约定：
- 具体模型适配器模块名来自 cfg.models[*].adapter，例如 "glm4v_flash"；
- 每个适配器模块需要实现一个工厂函数：
      create_model(name: str, **params) -> VLModel
"""

from __future__ import annotations

import time
import importlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 可选：设置 numpy / torch 随机种子（如果安装了）
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from src.core.cfg import Cfg, DataCfg, load_cfg
from src.core.io_utils import (
    read_jsonl,
    write_jsonl,
    write_csv,      # ★ 新增：写 csv
    setup_logger,
    ensure_dir,
    timestamp_str,
)
from src.core.schemas import (
    Sample,
    GroupedMission,
    PredictionRecord,
    image_from_base64,
    index_to_letter,
    letter_to_index,
)
from src.core import grouping, protocols
from src.adapters.base import VLModel


# ===================== 随机种子 & 模型加载 =====================

def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)


def _load_model_from_cfg(model_cfg, logger) -> VLModel:
    """
    根据 cfg.models[*] 加载一个 VLModel 实例。

    约定：
    - module = adapters.<adapter>
    - 该模块中必须提供函数 create_model(name: str, **params) -> VLModel
      例如在 adapters/glm_api.py 中：
          def create_model(name: str, **params):
              return GLM4VFlash(name=name, **params)
    """
    module_name = f"src.adapters.{model_cfg.adapter}"
    logger.info(f"Loading adapter module: {module_name}")
    module = importlib.import_module(module_name)

    if not hasattr(module, "create_model"):
        raise ImportError(
            f"Adapter module {module_name} must define a function `create_model(name: str, **params)`."
        )

    create_model = getattr(module, "create_model")
    model = create_model(name=model_cfg.name, **model_cfg.params)
    if not isinstance(model, VLModel):
        raise TypeError(
            f"Adapter {module_name}.create_model must return a VLModel instance, "
            f"but got {type(model)}"
        )
    logger.info(f"Loaded model: {model.name}")
    return model

def _correct_letter_from_sample(sample: Sample) -> str:
    """从样本里得到正确选项的字母（A/B/C/D）。"""
    if sample.correct_letter:
        return sample.correct_letter
    if sample.correct_index is not None:
        return chr(ord("A") + int(sample.correct_index))
    raise ValueError(f"Sample {sample.uid} 缺少 correct_letter / correct_index 信息")


def _wrong_letter_from_sample(sample: Sample, rng: random.Random) -> str:
    """从样本里随机挑一个错误选项的字母。"""
    n = sample.n_choices
    letters = [chr(ord("A") + i) for i in range(n)]
    correct = _correct_letter_from_sample(sample)
    wrong_letters = [L for L in letters if L != correct]
    if not wrong_letters:
        raise ValueError(f"Sample {sample.uid} 没有错误选项可选")
    return rng.choice(wrong_letters)

# ===================== 数据加载 & 转换 =====================

def _rows_to_samples(rows: List[Dict[str, Any]], data_cfg: DataCfg) -> List[Sample]:
    """
    将 jsonl 中的原始行转换为 Sample 列表。

    约定（尽量兼容）：
    - uid:        优先用 row["uid"] / row["id"]，否则自动生成 "sample_<idx>"；
    - image_id:   优先用 row["image_id"] / row["img_id"] / row["image"]；
    - image_b64:  优先 data_cfg.image_key，其次 "image_b64"；
    - question:   优先 data_cfg.text_key，其次 "question"；
    - choices:    优先 "choices"，其次 "options"、"multiple_choice_targets"；
                  若是 dict (如 {"A": "...", "B": "..."} )，按 key 排序为列表；
    - correct_letter:
          解析顺序：
            1) row["correct_letter"] 若存在；
            2) row[data_cfg.label_key]：
                 - 若是 list -> 取 argmax 作为正确答案索引；
                 - 若是 int/float -> 视为索引（默认 0-based）；
                 - 若是 str -> 若为数字字符串则视为索引，否则当作字母；
    - task_type:  可选字段 "task_type"；
    - meta:       将 row["meta"] 展开为顶层，再加上未使用的字段，便于 grouping 按 group_key / image_rel 分组。
    """
    samples: List[Sample] = []

    for idx, row in enumerate(rows):
        # ---------- 基本字段 ----------
        uid = row.get("uid") or row.get("id") or f"sample_{idx}"

        image_id = row.get("image_id") or row.get("img_id") or row.get("image")

        # image_b64
        if data_cfg.image_key in row:
            image_b64 = row.get(data_cfg.image_key)
        else:
            image_b64 = row.get("image_b64")

        # question
        question = row.get(data_cfg.text_key) or row.get("question") or ""

        # ---------- choices ----------
        raw_choices = (
            row.get("choices")
            or row.get("options")
            or row.get("multiple_choice_targets")
            or []
        )
        if isinstance(raw_choices, dict):
            choices = [raw_choices[k] for k in sorted(raw_choices.keys())]
        else:
            choices = list(raw_choices)

        # ---------- 正确答案 ----------
        raw_label = row.get("correct_letter")
        if raw_label is None and data_cfg.label_key in row:
            raw_label = row[data_cfg.label_key]

        correct_letter = ""

        if isinstance(raw_label, str):
            s = raw_label.strip()
            if s.isdigit():
                idx_int = int(s)
                correct_letter = index_to_letter(idx_int)
            else:
                # 直接当作字母
                correct_letter = s.upper()
        elif isinstance(raw_label, (int, float)):
            idx_int = int(raw_label)
            correct_letter = index_to_letter(idx_int)
        elif isinstance(raw_label, list):
            # 形如 [0, 0, 1, 0] -> 取 argmax
            if raw_label:
                try:
                    best_idx = max(range(len(raw_label)), key=lambda i: raw_label[i])
                    correct_letter = index_to_letter(int(best_idx))
                except Exception:
                    correct_letter = ""
        else:
            correct_letter = ""

        task_type = row.get("task_type") or None

        # ---------- meta：展开 row["meta"] 并合并其余字段 ----------
        raw_meta = row.get("meta") or {}
        if not isinstance(raw_meta, dict):
            raw_meta = {}

        used_keys = {
            "uid",
            "id",
            "image_id",
            "img_id",
            "image",
            data_cfg.image_key,
            "image_b64",
            data_cfg.text_key,
            "question",
            "choices",
            "options",
            "multiple_choice_targets",
            "correct_letter",
            data_cfg.label_key,
            "task_type",
            "meta",
        }
        extra_meta = {k: v for k, v in row.items() if k not in used_keys}

        meta = {
            **raw_meta,   # 这里会包含 group_key / image_rel / index_in_group 等关键信息
            **extra_meta,
        }

        samples.append(
            Sample(
                uid=str(uid),
                image_id=str(image_id) if image_id is not None else None,
                image_b64=image_b64,
                question=str(question),
                choices=[str(c) for c in choices],
                correct_letter=str(correct_letter),
                task_type=str(task_type) if task_type is not None else None,
                meta=meta,
            )
        )

    return samples




# ===================== 答案解析 =====================

def extract_answer(model_answer: str) -> str:
    """
    从模型输出中提取选项字母：
    - 模型现在被要求只输出单个大写字母（A/B/C/D）；
    - 但为稳妥起见，仍然在全文中寻找第一个 A/B/C/D；
    - 找不到则返回 ""。
    """
    if not model_answer:
        return ""

    # 统一成大写
    s = str(model_answer).upper()

    # 先找“独立的 A/B/C/D”（前后是非字母或边界）
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)

    # 再退一步：找括号里的 (A)/(B)/(C)/(D)
    m = re.search(r"\(([ABCD])\)", s)
    if m:
        return m.group(1)

    # 再退一步：从左到右扫第一个 A/B/C/D
    for ch in s:
        if ch in "ABCD":
            return ch

    return ""


# ===================== GroupedMission 构造 =====================

def _build_grouped_missions_for_protocol(
    cfg: Cfg,
    mission_units: List[grouping.MissionUnit],
    protocol: str,
    rng: random.Random,
) -> List[GroupedMission]:
    """
    给定一组 MissionUnit（history + test），为“某种协议”构造对应的 GroupedMission 列表。

    - raw:           不使用群体信息，直接 build_prompt_raw；
    - correct_guidance / wrong_guidance:
        群体选择通过 grouping.choose_group_letter_for_test 决定，
        具体正确/错误由 cfg.grouping.group_correct_ratio 控制；
    - trust / doubt:
        使用 BENCHFORM 风格的历史 + 当前问答：
          history_rounds = [(hq, hchoices, h_group_L, h_you_L), ...]
        再交给 protocols.build_prompt_trust_or_doubt 生成最终 prompt。
    """
    protocol = protocol.lower()
    missions: List[GroupedMission] = []

    for mu in mission_units:
        test = mu.test
        n_choices = test.n_choices
        choice_letters = [chr(ord("A") + i) for i in range(n_choices)]
        choice_pairs = list(zip(choice_letters, test.choices))

        # y_true & true_letter
        y_true = test.correct_index
        true_letter = test.correct_letter

        # 正确选项文本（方便落盘）
        correct_idx = test.correct_index
        if 0 <= correct_idx < test.n_choices:
            correct_text = test.choices[correct_idx]
        else:
            correct_text = None

        # ---------- 各协议分支 ----------

        if protocol == "raw":
            prompt = protocols.build_prompt_raw(test.question, choice_pairs)
            group_choice_letter = ""
            history_rounds: List[Any] = []


        elif protocol == "correct_guidance":
            # 强制群体选择正确答案
            group_choice_letter = _correct_letter_from_sample(test)
            prompt = protocols.build_prompt_guidance(
                test.question,
                choice_pairs,
                group_choice_letter,
                rng=rng,
            )
            history_rounds = []
        elif protocol == "wrong_guidance":
            # 强制群体选择错误答案
            group_choice_letter = _wrong_letter_from_sample(test, rng)
            prompt = protocols.build_prompt_guidance(
                test.question,
                choice_pairs,
                group_choice_letter,
                rng=rng,

            )

            history_rounds = []


        elif protocol == "trust":

            # ---------- TRUST: 历史群体全对，当前群体故意错 ----------

            history_rounds = []

            for h in mu.history:
                # 历史题目的选项

                h_letters = [chr(ord("A") + i) for i in range(h.n_choices)]

                h_choice_pairs = list(zip(h_letters, h.choices))

                # 群体一直选正确答案

                h_group_L = _correct_letter_from_sample(h)

                # 你自己的历史回答也设为正确答案

                h_you_L = _correct_letter_from_sample(h)

                history_rounds.append(

                    (h.question, h_choice_pairs, h_group_L, h_you_L)

                )

            # 当前题：群体这次故意选错

            group_choice_letter = _wrong_letter_from_sample(test, rng)

            prompt = protocols.build_prompt_trust_or_doubt(

                protocol="trust",

                current_question=test.question,

                current_choices=choice_pairs,

                history_rounds=history_rounds,

                current_group_choice_letter=group_choice_letter,

                rng=rng,

            )


        elif protocol == "doubt":

            # ---------- DOUBT: 历史群体全错，当前群体正确 ----------

            history_rounds = []

            for h in mu.history:
                h_letters = [chr(ord("A") + i) for i in range(h.n_choices)]

                h_choice_pairs = list(zip(h_letters, h.choices))

                # 群体一直选错误答案

                h_group_L = _wrong_letter_from_sample(h, rng)

                # 你自己的历史回答一直选正确答案

                h_you_L = _correct_letter_from_sample(h)

                history_rounds.append(

                    (h.question, h_choice_pairs, h_group_L, h_you_L)

                )

            # 当前题：群体这次选正确答案

            group_choice_letter = _correct_letter_from_sample(test)

            prompt = protocols.build_prompt_trust_or_doubt(

                protocol="doubt",

                current_question=test.question,

                current_choices=choice_pairs,

                history_rounds=history_rounds,

                current_group_choice_letter=group_choice_letter,

                rng=rng,

            )

        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        # ---------- 组装 GroupedMission ----------

        mission_id = f"{mu.group_key}#{test.uid}#{protocol}"
        gm = GroupedMission(
            mission_id=mission_id,
            protocol=protocol,
            prompt=prompt,
            image_b64=test.image_b64,
            y_true=y_true,
            true_letter=true_letter,
            group_key=mu.group_key,
            source_uids=[s.uid for s in mu.history] + [test.uid],
            meta={
                "task_type": test.task_type,
                "group_choice_letter": group_choice_letter,
                "context_k": len(mu.history),
                # 把原始问题、选项和正确答案放进 meta
                "question": test.question,
                "choices": test.choices,
                "correct_letter": test.correct_letter,
                "correct_text": correct_text,
            },
        )
        missions.append(gm)

    return missions


# ===================== 主流程 =====================

def run_eval(config_path: str) -> None:
    """
    单次实验入口函数：给脚本 run_experiment.py 调用。
    """
    cfg = load_cfg(config_path)

    # 日志 & 随机种子
    exp_tag = f"{cfg.exp_name}_{timestamp_str()}"
    log_file = (
        str(Path(cfg.save.log_dir) / f"{exp_tag}.log")
        if cfg.save.log_dir
        else None
    )
    logger = setup_logger(name="eval_runner", log_file=log_file)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Experiment tag: {exp_tag}")
    _set_random_seed(cfg.seed)

    # 1. 读数据 -> Sample 列表
    logger.info(f"Loading data from {cfg.data.path}")
    raw_rows = read_jsonl(cfg.data.path)
    samples = _rows_to_samples(raw_rows, cfg.data)
    logger.info(f"Loaded {len(samples)} samples")

    # 2. 构造 3+1 任务单元
    mission_units = grouping.build_mission_units(
        samples, context_k=cfg.grouping.context_k
    )
    logger.info(
        f"Built {len(mission_units)} mission units "
        f"(context_k={cfg.grouping.context_k})"
    )

    if not mission_units:
        logger.warning("No mission units constructed. Check your data/grouping config.")
        return

    # 3. 按模型依次评测
    runs_dir = Path(cfg.save.dir) / "runs" / cfg.exp_name
    ensure_dir(runs_dir)

    for model_cfg in cfg.models:
        logger.info(f"=== Evaluating model: {model_cfg.name} ===")
        model = _load_model_from_cfg(model_cfg, logger)
        rng = random.Random(cfg.seed)  # 每个模型独立但可复现

        all_pred_records: List[PredictionRecord] = []

        for protocol in cfg.protocols:
            logger.info(f"--- Protocol: {protocol} ---")
            missions = _build_grouped_missions_for_protocol(
                cfg, mission_units, protocol, rng
            )
            logger.info(f"Built {len(missions)} GroupedMissions for protocol={protocol}")

            correct_cnt = 0
            total_cnt = 0
            failed_cnt = 0  # 新增：记录被 API 拒绝 / 失败的个数

            # ===== 并行执行所有 missions =====

            def _run_one_mission(gm):
                """子线程里跑：解码图片 + 调模型 + 解析答案"""
                # 准备图像 (保持原样)
                if gm.image_b64:
                    try:
                        img_local = image_from_base64(gm.image_b64)
                    except Exception as e:
                        err = f"Decode image failed for mission {gm.mission_id}: {e}"
                        logger.error(err)
                        # 注意：这里返回时，output_text 设为空字符串或错误提示
                        return gm, "", err, "", -1
                else:
                    img_local = None

                # ================= 修正开始 =================
                # 1. 在循环外初始化变量，防止 UnboundLocalError
                output_text_local = ""
                error_msg_local: Optional[str] = None

                max_retries = 3

                for attempt in range(max_retries):
                    try:
                        # 尝试调用模型
                        output_text_local = model.predict(img_local, gm.prompt)
                        # 成功则清空错误信息并跳出循环
                        error_msg_local = None
                        break
                    except Exception as e:
                        error_msg_local = str(e)
                        is_last_attempt = (attempt == max_retries - 1)

                        if not is_last_attempt:
                            # 失败但还有机会，等待后重试
                            time.sleep(2 * (attempt + 1))
                        else:
                            # 最后一次尝试也失败了
                            logger.error(f"Mission {gm.mission_id} failed after {max_retries} attempts: {e}")
                            # ★★★ 关键点：最后一次失败时，必须给 output_text_local 赋值，否则后面会报错
                            output_text_local = f"[ERROR] {e}"
                # ================= 修正结束 =================

                # 解析答案 (此时 output_text_local 一定有值了)
                pred_letter_local = extract_answer(output_text_local)
                y_pred_local = letter_to_index(pred_letter_local)

                return gm, output_text_local, error_msg_local, pred_letter_local, y_pred_local

            max_workers = 4  # 可以根据自己机器和限速调，比如 4~6
            logger.info(
                f"Running {len(missions)} missions for protocol={protocol} "
                f"with {max_workers} worker threads..."
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run_one_mission, gm) for gm in missions]

                for fut in as_completed(futures):
                    gm, output_text, error_msg, pred_letter, y_pred = fut.result()

                    # 记录结果
                    record = PredictionRecord(
                        mission_id=gm.mission_id,
                        protocol=gm.protocol,
                        model_name=model_cfg.name,
                        inputs=gm.prompt,  # 包含协议的完整 prompt
                        outputs=output_text,  # 模型原始输出
                        y_true=gm.y_true,
                        y_pred=y_pred,
                        true_letter=gm.true_letter,
                        pred_letter=pred_letter,
                        image_b64=gm.image_b64,
                        meta={
                            "group_key": gm.group_key,
                            "task_type": gm.meta.get("task_type"),
                            "group_choice_letter": gm.meta.get("group_choice_letter"),
                            "context_k": gm.meta.get("context_k"),
                            "question": gm.meta.get("question"),
                            "choices": gm.meta.get("choices"),
                            "correct_letter": gm.meta.get("correct_letter"),
                            "correct_text": gm.meta.get("correct_text"),
                            "error": error_msg,
                        },
                    )
                    all_pred_records.append(record)

                    # 在线统计 Accuracy：
                    # 只对「没有 error 且 成功解析出选项字母」的样本计数
                    if (
                            error_msg is None
                            and gm.y_true is not None
                            and gm.y_true >= 0
                            and y_pred is not None
                            and y_pred >= 0
                    ):
                        total_cnt += 1
                        if record.is_correct:
                            correct_cnt += 1

                    # 统计失败次数
                    if error_msg is not None:
                        failed_cnt += 1

            acc = (correct_cnt / total_cnt) if total_cnt > 0 else 0.0
            logger.info(
                f"[{model_cfg.name}][{protocol}] "
                f"Accuracy = {acc:.4f} ({correct_cnt}/{total_cnt}), "
                f"failed={failed_cnt}"
            )

        # 4. 保存本模型的所有预测记录
        run_id = f"{model_cfg.name}_{exp_tag}"
        # ★ 改动点：用 to_legacy_row()（不含 image_base64），并同时写 jsonl + csv
        rows = [r.to_legacy_row() for r in all_pred_records]

        out_jsonl = runs_dir / f"{run_id}.jsonl"
        logger.info(f"Saving prediction records to {out_jsonl}")
        write_jsonl(out_jsonl, rows)

        out_csv = runs_dir / f"{run_id}.csv"
        logger.info(f"Saving prediction records (csv) to {out_csv}")
        write_csv(out_csv, rows)

        logger.info(
            f"Done saving {len(all_pred_records)} records for {model_cfg.name}"
        )

    logger.info("All models finished.")


# ===================== 可选：CLI 入口 =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one conformity experiment (multi-model, multi-protocol, 3+1)."
    )
    parser.add_argument("config", help="Path to yaml config file.")
    args = parser.parse_args()
    run_eval(args.config)
