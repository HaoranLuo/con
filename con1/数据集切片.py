# -*- coding: utf-8 -*-
"""
数据集切片（分片版，按“图片分组”切；保证3+1完整、不打断）

- 从 experiment_ready(3).jsonl 按组读取（同一张图的一组样本，通常=take_per_image 条）
- 按 groups_per_shard 或 max_shard_bytes 做分片
- 输出为多份 JSONL 文件：每行一个样本，方便 eval_runner 的 read_jsonl 直接读取
- 可选：仅取前 top_groups 组（做快速小样）
"""

import os
import json
import argparse
import math
import random


def iter_jsonl(path):
    """逐行读取 jsonl 文件，每行反序列化为一个 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_by_image(src_jsonl, prefer_key="group_key"):
    """
    将 experiment_ready(3).jsonl 中的样本按“图像/分组键”聚合：
      返回 {group_key: [items]}；
      若 meta.group_key 不存在，则退化使用 meta.image_rel。
    """
    groups = {}
    for obj in iter_jsonl(src_jsonl):
        meta = obj.get("meta", {}) or {}
        gk = meta.get(prefer_key)
        if not gk:
            gk = meta.get("image_rel") or "unknown"
        groups.setdefault(gk, []).append(obj)
    return groups


def shard_groups(
    groups,
    out_dir,
    groups_per_shard=100,
    max_shard_bytes=None,
    seed=42,
    top_groups=None,
):
    """
    将按 group_key 聚合后的 dict 切成多片；每片写成一个 JSONL 文件：
      - 每片文件名形如：val_data_shard_0001.jsonl
      - 文件内容：一行一个样本（与原 experiment_ready(3).jsonl 的单行结构一致）

    参数：
      - groups_per_shard：每片包含的组数（优先使用）
      - max_shard_bytes：若设置，则在不超过该体积的前提下尽量多装几组（近似估计）
      - top_groups：仅取前多少组（用于抽样/快速测试）
    """
    keys = list(groups.keys())
    rnd = random.Random(seed)
    # 为了可复现，先按键排序；如果以后想随机，可以加 rnd.shuffle(keys)
    keys.sort()
    if top_groups is not None:
        keys = keys[: int(top_groups)]

    shards = []
    cur, cur_size_est = [], 0

    def flush_shard(shard_items, idx):
        """将当前收集的样本写成 JSONL 文件。"""
        shard_name = f"val_data_shard_{idx:04d}.jsonl"
        shard_path = os.path.join(out_dir, shard_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(shard_path, "w", encoding="utf-8") as f:
            for obj in shard_items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return shard_path

    shard_idx = 1
    for k in keys:
        g = groups[k]
        # 估算体积：把该组序列化一次（粗略估算）
        g_bytes = len(json.dumps(g, ensure_ascii=False).encode("utf-8"))

        # 条件1：组数到达上限 -> 刷片
        cond_groups = (groups_per_shard is not None and len(cur) >= groups_per_shard)
        # 条件2：体积预估会超限 -> 刷片
        cond_bytes = (
            max_shard_bytes is not None
            and (cur_size_est + g_bytes) > max_shard_bytes
        )

        if cur and (cond_groups or cond_bytes):
            path = flush_shard(cur, shard_idx)
            shards.append(path)
            shard_idx += 1
            cur, cur_size_est = [], 0

        cur.extend(g)
        cur_size_est += g_bytes

    if cur:
        path = flush_shard(cur, shard_idx)
        shards.append(path)

    return shards


def main():
    ap = argparse.ArgumentParser(
        description="按图片分组切片 experiment_ready(3).jsonl，输出 JSONL 分片"
    )
    ap.add_argument(
        "--src",
        default=r"E:\处理\3题\experiment_ready.jsonl",
        help="输入的 experiment_ready(3).jsonl 路径",
    )
    ap.add_argument(
        "--dst_dir",
        default=r"C:\Users\骆浩然\Desktop\project\data\seed_h\shards",
        help="输出分片文件夹",
    )
    ap.add_argument(
        "--groups_per_shard",
        type=int,
        default=200,
        help="每片包含的组数（优先级高）",
    )
    ap.add_argument(
        "--max_shard_bytes",
        type=int,
        default=None,
        help="每片最大体积，单位字节（可选）",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--top_groups",
        type=int,
        default=None,
        help="仅取前 top 组，做快速小样（可选）",
    )
    # ⭐ 新增：最小 group 大小（比如 4 = 3+1）
    ap.add_argument(
        "--min_group_size",
        type=int,
        default=3,
        help="只保留样本数 >= min_group_size 的图像分组",
    )

    args = ap.parse_args()

    # 1) 先按图像分组
    all_groups = group_by_image(args.src)
    print(f"[Info] 原始共 {len(all_groups)} 个图像分组")

    # 2) 再过滤掉样本数太少的 group
    groups = {
        k: v for k, v in all_groups.items() if len(v) >= args.min_group_size
    }
    print(
        f"[Info] 过滤掉样本数 < {args.min_group_size} 的分组后，剩余 {len(groups)} 个分组"
    )

    # 3) 再做切片
    shards = shard_groups(
        groups,
        out_dir=args.dst_dir,
        groups_per_shard=args.groups_per_shard,
        max_shard_bytes=args.max_shard_bytes,
        seed=args.seed,
        top_groups=args.top_groups,
    )
    print("[OK] 生成 JSONL 分片：")
    for p in shards:
        print("  -", p)
    print("\n提示：在 configs/*.yaml 里，把 data.path 指向某个分片，例如：")
    print('  data:')
    print('    path: "../data/seed_h/shards/val_data_shard_0001.jsonl"')
    print("  然后用 run_experiment.py 跑这个小分片即可。")



if __name__ == "__main__":
    main()
