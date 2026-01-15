"""
一键生成三件套（v3.5 no-zip，方案B：以图片来源根目录为命名空间）
1) questions_one_image_multiq.jsonl —— 一图多题问题集
2) images/                          —— 规范化拷贝的图片目录
3) experiment_ready(3).jsonl           —— 含 image_base64 的实验输入

改动亮点（v3.5）：
- extract_choices(q): 统一抽取选项（分散键/列表键/题干内嵌）
- onehot_from_answer_letter: 鲁棒解析答案（字母/数字/文本近似）
- ✅ 修复：支持以“完整相对路径（含/不含扩展名）”作为 data_id 的图片精确匹配
- ✅ 新增：data_id 若指向目录（如 task14/image/test/5），递归在目录内匹配图片；再退化为全库“目录前缀”匹配
- ✅ 新增：drops_by_subdir_reason.tsv / drops_by_subdir_examples.tsv（按规范化次级目录 × 原因的统计与示例）
- ✅ 新增：meta.group_key（本次新增，保证同一张图的问题能稳定分组）
"""

import os, re, json, time, shutil, argparse, collections, base64, random, difflib
from pathlib import PurePosixPath, Path
from collections import defaultdict

# ========= 默认路径 =========
DEFAULT_QUESTIONS_JSON = r"E:\数据集\SEED-Bench-H.json"
DEFAULT_IMAGES_SRC     = [
    r"E:\数据集\SEED-Bench-2-image",
    r"E:\数据集\cc3m-image",
    r"E:\数据集\SEED-Bench-H-data",
]
DEFAULT_OUT_DIR        = r"E:\处理\3题"

# ========= 常量 =========
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff")
REL_IMG_PREFIX = "images"

# ⭐ 新增：每个“数据集来源”最多保留多少张图片（按 images/ 后的第一层目录计数）
# 例如：
#   images/cc3m-image/...
#   images/SEED-Bench-H-data/...
#   images/SEED-Bench-2-image/...
# 分别最多保留 MAX_PER_SOURCE 张图片及对应问题
MAX_PER_SOURCE = 200

MULTI_IMG_PATTERNS = [
    "<img>", "option images", "corresponding depth images",
    "select correct corresponding image", "the option images are"
]

# 标准子集标识（用于路径规范化的锚点；全部使用规范化小写token）
STANDARD_SUBSETS = {"seed-bench-2-image", "seed-bench-h-data", "cc"}

# ========= 小工具 =========
def ensure_dirs(p): os.makedirs(p, exist_ok=True)

def normalize_noext(path: str) -> str:
    p = path.replace("\\", "/").lower()
    base = os.path.basename(p)
    if "." in base: p = p[: p.rfind(".")]
    return p

def load_questions(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("JSON 结构不含 'questions'")
    return data["questions"]

def collect_multiq(questions, min_repeat=2):
    """
    将同一图片的题目聚合。这里对 data_id 做一次规范化（小写 + / 分隔），减少大小写/分隔符差异导致的错配。
    """
    grouped = defaultdict(list)
    for q in questions:
        did = q.get("data_id", None)
        if did is None:
            continue
        if isinstance(did, (list, tuple, set)):
            dids = [str(x).replace("\\","/").lower() for x in did if x is not None]
        else:
            dids = [str(did).replace("\\","/").lower()]
        for one in dids:
            if one:
                grouped[one].append(q)
    multiq_ids = {did for did, lst in grouped.items() if len(lst) >= min_repeat}
    return multiq_ids, grouped

# ========= 关键修复：选项抽取 =========
CHOICE_KEY_PATTERNS = [
    r'^choice_?([a-z])$',           # choice_a / choicea
    r'^option_?([1-9])$',           # option1 / option_1
    r'^label_?([a-d])$',            # label_a
    r'^ans_?([a-d])$',              # ans_a
]

def _norm_opt_text(s: str) -> str:
    return (s or "").strip()

def _extract_from_question_text(qtext: str):
    """从题干解析 (A) ... (B) ... (C) ... (D) ..."""
    t = (qtext or "").replace("\n", " ")
    t = t.replace("（", "(").replace("）", ")")
    parts = re.split(r'\(([A-Fa-f])\)', t)
    opts_map = {}
    i = 1
    while i + 1 < len(parts):
        label = parts[i].upper()
        text = parts[i+1]
        opts_map[label] = _norm_opt_text(text)
        i += 2
    out = []
    for k in ["A","B","C","D","E","F"]:
        if k in opts_map:
            out.append(opts_map[k])
    return out

def extract_choices(qobj: dict):
    """
    1) 列表键：choices / options / mc_options
    2) 分散键：choice_a / choiceA / option_1 / label_a / ans_a ...
    3) 题干内嵌：(A) ... (B) ... (C) ... (D) ...
    """
    # 1) 列表键
    for key in ["choices", "options", "mc_options"]:
        val = qobj.get(key)
        if isinstance(val, list) and val:
            return [_norm_opt_text(x) for x in val]

    # 2) 分散键（统一小写）
    low = {str(k).strip().lower(): v for k, v in qobj.items() if v is not None}
    bucket = {}
    for k, v in low.items():
        for pat in CHOICE_KEY_PATTERNS:
            m = re.match(pat, k)
            if m:
                tag = m.group(1)
                if tag.isdigit():  # 数字 → ABCD
                    idx = int(tag) - 1
                    if 0 <= idx < 26:
                        tag = chr(ord('a') + idx)
                bucket[tag.upper()] = _norm_opt_text(str(v))
                break
    if bucket:
        out = []
        for lab in ["A","B","C","D","E","F"]:
            if lab in bucket:
                out.append(bucket[lab])
        if out:
            return out

    # 3) 题干内嵌
    qtext = qobj.get("question") or qobj.get("prompt") or ""
    out = _extract_from_question_text(qtext)
    if out:
        return out

    return []

# ========= 关键修复：更鲁棒的答案解析 =========
def _norm_txt(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r'[\s\.\,\:\;\!\?\(\)\[\]\{\}\"\'`~•·…]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def onehot_from_answer_letter(ans_raw, choices):
    """
    1) 字母: A/B/C/D（忽略括号/多余字符）
    2) 数字: 1/2/3/4 或 'option 2'
    3) 文本: 与选项文本近似（difflib ≥ 0.90）
    返回: (scores, idx)
    """
    opts = [str(c or "").strip() for c in (choices or [])]
    n = len(opts)
    scores = [0] * n
    if n == 0:
        return scores, None

    s = str(ans_raw or "").strip()

    # ① 抽取 A-D 字母
    m = re.search(r'([A-Da-d])', s)
    if m:
        idx = ord(m.group(1).upper()) - ord('A')
        if 0 <= idx < n:
            scores[idx] = 1
            return scores, idx

    # ② 抽取数字索引 (1-based)
    m = re.search(r'(?<!\d)([1-9])', s)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            scores[idx] = 1
            return scores, idx

    # ③ 文本近似匹配
    s_norm = _norm_txt(s)
    s_norm = re.sub(r'^\([a-d]\)\s*', '', s_norm)  # 去掉 "(c) "
    norm_map = {_norm_txt(opt): i for i, opt in enumerate(opts)}
    if s_norm in norm_map:
        idx = norm_map[s_norm]
        scores[idx] = 1
        return scores, idx

    if s_norm:
        ratios = [(difflib.SequenceMatcher(None, s_norm, _norm_txt(opt)).ratio(), i) for i, opt in enumerate(opts)]
        best_ratio, best_i = max(ratios, key=lambda x: x[0])
        if best_ratio >= 0.90:
            scores[best_i] = 1
            return scores, best_i

    return scores, None

def build_parsed_inputs(rel_img_path, question_text, options):
    lines = [f"Image: {rel_img_path}", f"Question: {question_text}"]
    for i, o in enumerate(options):
        lines.append(f"({chr(ord('A')+i)}) {o}")
    return "\n".join(lines)

def encode_image_base64(abs_path: str) -> str:
    ext = os.path.splitext(abs_path)[1].lower().lstrip(".")
    if ext == "jpg": ext = "jpeg"
    mime = f"image/{ext}" if ext in {"jpeg","png","webp","bmp","gif","tiff"} else "application/octet-stream"
    with open(abs_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def is_multi_image_text(text: str) -> bool:
    tl = (text or "").lower()
    return any(p in tl for p in MULTI_IMG_PATTERNS)

# ========= 目录 & 路径规范化 =========
def _sanitize_dirname(name: str | None) -> str:
    if not name:
        return "unknown"
    s = str(name).strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower())

def canonicalize_subrel_dir(namespace_hint: str | None, rel_from_root: str, rewrite_log=None) -> str:
    """
    规则：
    - 去掉开头一次 images/image/imgs
    - 若路径中出现标准子集名（seed-bench-2-image / seed-bench-h-data / cc），锚定第一次出现，仅保留该处
    - 否则：以 namespace_hint 为顶层命名空间；若首段与 namespace_hint 同名，去一次
    """
    rel_subdir = os.path.dirname(rel_from_root).replace("\\", "/")
    parts = [p for p in rel_subdir.split("/") if p]

    if parts and parts[0].lower() in ("images", "image", "imgs"):
        parts = parts[1:]

    normed = [_norm_token(p) for p in parts]
    anchor_idx = next((i for i, t in enumerate(normed) if t in STANDARD_SUBSETS), -1)

    ns_dir_raw = _sanitize_dirname(namespace_hint) if namespace_hint else "unknown"
    ns_dir = ns_dir_raw
    ns_norm = _norm_token(ns_dir)

    before = "/".join(parts)

    if anchor_idx >= 0:
        out = []
        seen_subset = False
        for j in range(anchor_idx, len(parts)):
            tok = normed[j]
            if tok in STANDARD_SUBSETS:
                if seen_subset:
                    continue
                seen_subset = True
            out.append(parts[j])
        subrel = "/".join(out) if out else ns_dir
    else:
        if parts and _norm_token(parts[0]) == ns_norm:
            parts = parts[1:]
        subrel = "/".join([ns_dir] + parts) if parts else ns_dir

    if rewrite_log is not None and before != subrel:
        rewrite_log.write(before + "  -->  " + subrel + "\n")
    return subrel

# ========= 全局索引（纯目录版） =========
def build_global_index(root_dirs):
    """
    返回：
      id_map: key -> [(root_dir, rel_path), ...]
      meta  : (root_dir, rel_path) -> (is_latex, depth, strlen, full_lower)
    """
    id_map = defaultdict(list)
    meta = {}

    def add_record(root_dir, rel_path):
        disp = rel_path.replace("\\", "/")
        full_lower = disp.lower()
        noext = normalize_noext(disp)
        parts = PurePosixPath(noext).parts
        is_latex = 1 if "latex_code_image" in full_lower else 0
        meta[(root_dir, rel_path)] = (is_latex, len(parts), len(noext), full_lower)

        keys = set()
        # 现有键：文件名无扩展 / 上一级_文件名 / 数字组合
        if parts:
            keys.add(parts[-1])
        if len(parts) >= 2:
            keys.add(parts[-2] + "_" + parts[-1])
        nums = re.findall(r"\d+", noext)
        if len(nums) >= 1: keys.add(nums[-1])
        if len(nums) >= 2: keys.add(nums[-2] + "_" + nums[-1])
        if len(nums) >= 3: keys.add(nums[-3] + "_" + nums[-2] + "_" + nums[-1])

        # ✅ 路径型键（带/不带扩展名）
        keys.add(full_lower)          # 例：task11/landmark_image/00c08b...jpg
        keys.add(noext.lower())       # 例：task11/landmark_image/00c08b...

        for k in keys:
            id_map[k].append((root_dir, rel_path))

    for rd in root_dirs:
        rd = os.path.abspath(rd)
        if not os.path.isdir(rd):
            print(f"[Warn] 非目录，跳过：{rd}")
            continue
        for r, _d, files in os.walk(rd):
            for fn in files:
                low = fn.lower()
                if any(low.endswith(ext) for ext in IMG_EXTS):
                    rel = os.path.relpath(os.path.join(r, fn), rd)
                    add_record(rd, rel)
    print(f"[Index] 完成：共 {sum(len(v) for v in id_map.values())} 条键映射")
    return id_map, meta

# ========= 辅助：从 data_id 推测“标准化次级目录（subdir）” =========
def _subdir_from_did(did: str, namespace_hint: str | None) -> str:
    """
    用 data_id 直接推一个“规范化次级目录”。当图片还没复制时，用它来做日志聚合的目录键。
    """
    did = (did or "").replace("\\", "/")
    # 若 data_id 包含标准子集锚点，从第一次出现处开始
    parts = [p for p in did.split("/") if p]
    normed = [_norm_token(p) for p in parts]
    anchor_idx = next((i for i, t in enumerate(normed) if t in STANDARD_SUBSETS), -1)
    if anchor_idx >= 0:
        parts = parts[anchor_idx:]
    else:
        # 否则用 namespace_hint + data_id 前若干层
        ns = _sanitize_dirname(namespace_hint) if namespace_hint else "unknown"
        parts = [ns] + parts[:-1]  # 去掉可能的文件名
    return "/".join([p for p in parts if p])

# ========= 查找候选（新增：目录式 did 的递归/前缀匹配） =========
def find_candidate(id_map, meta, data_id: str, _unused_data_source: str | None):
    """
    强化版检索：
    - 直接尝试 data_id 的完整相对路径（带扩展/不带扩展）
    - 同时尝试文件名（带扩展/不带扩展）
    - 兼容原有的数字切片键
    - ✅ 若仍无匹配：将 data_id 视为“目录前缀”，在索引中找以该前缀开头（或包含该段路径）的图片
    """
    did = str(data_id or "").strip().replace("\\", "/").lower()
    if not did:
        return None

    did_base = os.path.basename(did)                   # 00c08b....jpg 或 5
    did_base_noext = os.path.splitext(did_base)[0]     # 00c08b.... 或 5
    did_noext_path = normalize_noext(did)              # task11/landmark_image/00c08b.... 或 task14/image/test/5

    keys = {
        did,                   # 完整相对路径（带扩展）
        did_noext_path,        # 完整相对路径（无扩展）
        did_base,              # 纯文件名（带扩展）
        did_base_noext,        # 纯文件名（无扩展）
    }

    # 兼容原有：数字片段
    nums = re.findall(r"\d+", did)
    if len(nums) >= 1: keys.add(nums[-1])
    if len(nums) >= 2: keys.add(nums[-2] + "_" + nums[-1])
    if len(nums) >= 3: keys.add(nums[-3] + "_" + nums[-2] + "_" + nums[-1])

    # 汇总候选
    cands = []
    for k in keys:
        cands.extend(id_map.get(k, []))

    # ============== 新增：目录式 did 的前缀/包含匹配 ==============
    if not cands:
        # 视为目录：尝试在全库中以 did 为“路径前缀”
        did_prefix = did.strip("/")

        def _score_path_like(c):
            root_dir, rel_path = c
            rel_low = rel_path.replace("\\", "/").lower()
            hard = 1 if (rel_low.startswith(did_prefix + "/") or ("/" + did_prefix + "/") in ("/" + rel_low)) else 0
            # 次序依据：前缀强匹配 > 文件大小 > 最近修改
            try:
                stat = os.stat(os.path.join(root_dir, rel_path))
                size = stat.st_size
                mtime = stat.st_mtime
            except Exception:
                size, mtime = 0, 0.0
            return (hard, size, mtime)

        pool = list(meta.keys())  # 所有 (root_dir, rel_path)
        if pool:
            pool.sort(key=_score_path_like, reverse=True)
            # 首个有效即取
            top = pool[0]
            # 如果完全不匹配（hard=0）且 did 是非常短的片段，就放弃
            rel_low = top[1].replace("\\", "/").lower()
            hard_ok = rel_low.startswith(did_prefix + "/") or ("/" + did_prefix + "/") in ("/" + rel_low)
            if hard_ok:
                return top
            # 否则仍然返回 None（避免误配）
    # ==========================================================

    if not cands:
        return None

    # 优先：与 did 前缀更长、路径越“像”的候选
    def score(c):
        root_dir, rel_path = c
        rel_low = rel_path.replace("\\", "/").lower()
        # 最长公共前缀长度（越长越好），再按原元信息排序
        lcp = len(os.path.commonprefix([rel_low, did]))
        is_latex, depth, strlen, full_lower = meta[c]
        return (lcp, -is_latex, -depth, strlen)

    cands.sort(key=score, reverse=True)
    return cands[0]

# ========= 主流程 =========
def run(questions_json, images_sources, output_dir,
        min_repeat=2, strict_min_q=2,
        require_single_image_type=False,
        filter_multi_in_qset=False,
        take_per_image=4, seed=42, pick_strategy="first", write_group_tags=False,
        clean_out_images=False):

    rnd = random.Random(seed)
    t0 = time.time()
    ensure_dirs(output_dir)
    out_img_dir = os.path.join(output_dir, REL_IMG_PREFIX)

    if clean_out_images and os.path.isdir(out_img_dir):
        shutil.rmtree(out_img_dir)
    ensure_dirs(out_img_dir)

    qset_path = os.path.join(output_dir, "questions_one_image_multiq.jsonl")
    exp_ready_path = os.path.join(output_dir, "experiment_ready(3).jsonl")
    missing_log = os.path.join(output_dir, "missing_images.txt")
    stats_txt = os.path.join(output_dir, "stats.txt")
    rewrites_log = os.path.join(output_dir, "path_rewrites.txt")

    drop_missing_detail_log = os.path.join(output_dir, "drops_missing_after_qset.txt")
    drop_missing_by_dir_log = os.path.join(output_dir, "missing_by_dir.txt")
    copy_errors_log = os.path.join(output_dir, "copy_errors.txt")

    # 新增：更细的“按次级目录×原因”日志
    drops_by_subdir_reason = collections.Counter()
    drops_examples = collections.defaultdict(list)
    drops_by_subdir_reason_tsv = os.path.join(output_dir, "drops_by_subdir_reason.tsv")
    drops_examples_tsv = os.path.join(output_dir, "drops_by_subdir_examples.tsv")

    # 1) 读取题库 & 聚合
    questions = load_questions(questions_json)
    multiq_ids, grouped = collect_multiq(questions, min_repeat=min_repeat)
    print(f"[Step1] 题目总数={len(questions)}；data_id 重复(≥{min_repeat})的图片数={len(multiq_ids)}")

    # 2) 索引图片来源（纯目录）
    sources = [os.path.abspath(p) for p in (images_sources or []) if os.path.isdir(p)]
    abs_out = os.path.abspath(output_dir)
    sources = [s for s in sources if not os.path.abspath(s).startswith(abs_out)]
    id_map, meta = build_global_index(sources)  # 参见原实现
    # ------------------------------------------------------------------------

    kept_imgs, did_to_imgname, missing = set(), {}, []
    kept_q_lines = 0

    # 失败样本落盘（解析不到选项/答案）
    _no_choice_cap = 200
    _no_choice_dumped = 0
    _unonehot_cap = 200
    _unonehot_dumped = 0

    with open(qset_path, "w", encoding="utf-8") as qout, \
         open(copy_errors_log, "w", encoding="utf-8") as copylog, \
         open(rewrites_log, "w", encoding="utf-8") as rlog:

        for did in sorted(multiq_ids):
            # 找图：找不到则整组跳过
            ds_from_question = None
            cand = find_candidate(id_map, meta, did, ds_from_question)
            if cand is None:
                missing.append((did, "image_not_found"))
                # 细化：把 data_id 映射到可能的 subdir，记原因
                subdir = _subdir_from_did(did, namespace_hint=None)
                key = (subdir, "image_not_found")
                drops_by_subdir_reason[key] += 1
                if len(drops_examples[key]) < 5:
                    drops_examples[key].append({"data_id": did, "stage": "qset", "note": "find_candidate=None"})
                continue

            root_dir, rel_path = cand

            # 顶层命名空间 = 图片来源根目录名
            root_basename = os.path.basename(os.path.normpath(root_dir))
            subrel_dir = canonicalize_subrel_dir(root_basename, rel_path, rewrite_log=rlog)

            abs_subdir = os.path.join(out_img_dir, subrel_dir)
            ensure_dirs(abs_subdir)

            if did not in did_to_imgname:
                leaf_name = os.path.basename(rel_path)
                src_abs = os.path.join(root_dir, rel_path)
                dst_abs = os.path.join(abs_subdir, leaf_name)

                # 同名防冲突
                if os.path.exists(dst_abs):
                    root_name, ext = os.path.splitext(leaf_name)
                    dst_abs = os.path.join(abs_subdir, f"{did}{ext or ''}")

                try:
                    shutil.copy2(src_abs, dst_abs)
                except Exception as e:
                    copylog.write(f"{did}\tcopy_failed\t{src_abs}\t->\t{dst_abs}\t{e}\n")
                    key = (subrel_dir, "copy_failed")
                    drops_by_subdir_reason[key] += 1
                    if len(drops_examples[key]) < 5:
                        drops_examples[key].append({"data_id": did, "stage": "copy", "src": src_abs, "dst": dst_abs, "err": str(e)})
                    continue

                rel_subpath = os.path.join(subrel_dir, os.path.basename(dst_abs)).replace("\\", "/")
                did_to_imgname[did] = rel_subpath
                kept_imgs.add(rel_subpath)

            rel_img = os.path.join(REL_IMG_PREFIX, did_to_imgname[did]).replace("\\", "/")

            # 写问题集（QSET）：统一用 extract_choices
            for q in grouped[did]:
                if filter_multi_in_qset and is_multi_image_text(q.get("question","")):
                    key = (os.path.dirname(did_to_imgname[did]) or subrel_dir, "filtered_multi_image_in_qset")
                    drops_by_subdir_reason[key] += 1
                    if len(drops_examples[key]) < 5:
                        drops_examples[key].append({"data_id": did, "question_id": q.get("question_id"), "stage": "qset", "note": "is_multi_image_text"})
                    continue

                options = extract_choices(q)

                if not options and _no_choice_dumped < _no_choice_cap:
                    with open(os.path.join(output_dir, "answer_no_choices_samples.txt"), "a", encoding="utf-8") as ff:
                        ff.write(json.dumps({
                            "image_rel": rel_img,
                            "data_id": did,
                            "question_id": q.get("question_id"),
                            "question": (q.get("question","") or "")[:300]
                        }, ensure_ascii=False) + "\n")
                    _no_choice_dumped += 1
                    key = (os.path.dirname(did_to_imgname[did]) or subrel_dir, "no_choices")
                    drops_by_subdir_reason[key] += 1
                    if len(drops_examples[key]) < 5:
                        drops_examples[key].append({"data_id": did, "question_id": q.get("question_id"), "stage": "qset", "note": "no choices"})

                item = {
                    "data_id": did,
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "choices": options,
                    "answer": q.get("answer", ""),
                    "data_source": q.get("data_source"),
                    "level": q.get("level"),
                    "subpart": q.get("subpart"),
                    "version": q.get("version"),
                    "image_rel": rel_img,
                    "data_type": q.get("data_type"),
                    # ✅ 新增：稳定分组键
                    "meta": {"group_key": did}
                }
                qout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_q_lines += 1

    # 3) 生成 experiment_ready(3).jsonl（含 Base64）
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except:
                        pass

    groups = collections.defaultdict(list)
    missing_detail_records = []
    missing_by_dir_counter = collections.Counter()
    rnd = random.Random(seed)

    for obj in load_jsonl(qset_path):
        if is_multi_image_text(obj.get("question","")):
            # 已在 QSET 计数，这里跳过
            continue

        rel = obj.get("image_rel")
        if not rel:
            missing_detail_records.append((obj.get("data_id"), obj.get("question_id"), None, None, "empty_rel"))
            # 记录 subdir 原因
            did_guess = obj.get("data_id") or ""
            subdir = _subdir_from_did(did_guess, namespace_hint=None)
            key = (subdir, "empty_rel")
            drops_by_subdir_reason[key] += 1
            if len(drops_examples[key]) < 5:
                drops_examples[key].append({"data_id": did_guess, "question_id": obj.get("question_id"), "stage": "exp", "note": "empty_rel"})
            continue

        abs_img_path = os.path.join(output_dir, rel)
        if not os.path.isfile(abs_img_path):
            missing_detail_records.append(
                (obj.get("data_id"), obj.get("question_id"), rel, rel, "not_found")
            )
            rel_dir = os.path.dirname(rel).replace("\\", "/") or REL_IMG_PREFIX
            missing_by_dir_counter[rel_dir] += 1
            key = (rel_dir.replace(f"{REL_IMG_PREFIX}/", "", 1) if rel_dir.startswith(f"{REL_IMG_PREFIX}/") else rel_dir, "not_found")
            drops_by_subdir_reason[key] += 1
            if len(drops_examples[key]) < 5:
                drops_examples[key].append({"data_id": obj.get("data_id"), "question_id": obj.get("question_id"), "stage": "exp", "note": "image file missing"})
            continue

        if require_single_image_type and str((obj.get("data_type") or "")).strip().lower() != "single image":
            key = (os.path.dirname(rel).replace(f"{REL_IMG_PREFIX}/", "", 1), "not_single_image")
            drops_by_subdir_reason[key] += 1
            if len(drops_examples[key]) < 5:
                drops_examples[key].append({"data_id": obj.get("data_id"), "question_id": obj.get("question_id"), "stage": "exp", "note": "require_single_image_type"})
            continue

        groups[rel].append(obj)

    # ⭐ 新增：每个数据集来源的计数（在 experiment_ready 阶段限制每来源最多 MAX_PER_SOURCE 张图片）
    source_counter = collections.Counter()

    with open(os.path.join(output_dir, "experiment_ready(3).jsonl"), "w", encoding="utf-8") as g:
        for rel, items in groups.items():

            # ---- 新增：按“数据集来源”限制每来源最多 MAX_PER_SOURCE 张图片 ----
            # rel 形如：
            #   images/cc3m-image/100102_2541465287.jpg
            #   images/SEED-Bench-H-data/text_rich/380.png
            #   images/SEED-Bench-2-image/task12/png/122.png
            subpath = rel
            if rel.startswith(f"{REL_IMG_PREFIX}/"):
                subpath = rel[len(REL_IMG_PREFIX) + 1:]  # 去掉 "images/"
            source = subpath.split("/", 1)[0] if "/" in subpath else subpath

            if MAX_PER_SOURCE and source_counter[source] >= MAX_PER_SOURCE:
                # 记录被“来源限额”丢弃
                subdir_key = os.path.dirname(rel).replace(f"{REL_IMG_PREFIX}/", "", 1)
                key = (subdir_key, "per_source_limit")
                drops_by_subdir_reason[key] += 1
                if len(drops_examples[key]) < 5:
                    drops_examples[key].append({
                        "rel": rel,
                        "source": source,
                        "stage": "exp",
                        "note": "per_source_limit"
                    })
                continue
            # ---- 新增部分结束 ----

            # “题目数不足 → 整组丢弃”
            if len(items) < strict_min_q or len(items) < take_per_image:
                key = (os.path.dirname(rel).replace(f"{REL_IMG_PREFIX}/", "", 1), "too_few_questions")
                drops_by_subdir_reason[key] += 1
                if len(drops_examples[key]) < 5:
                    drops_examples[key].append({"rel": rel, "group_size": len(items), "need": max(strict_min_q, take_per_image), "stage": "exp"})
                continue

            picked = items[:take_per_image] if pick_strategy == "first" else rnd.sample(items, take_per_image)

            abs_img_path = os.path.join(output_dir, rel)
            if not os.path.isfile(abs_img_path):
                missing_detail_records.append(("<group_write>", None, rel, rel, "group_write_missing"))
                key = (os.path.dirname(rel).replace(f"{REL_IMG_PREFIX}/", "", 1), "group_write_missing")
                drops_by_subdir_reason[key] += 1
                if len(drops_examples[key]) < 5:
                    drops_examples[key].append({"rel": rel, "stage": "exp", "note": "group_write_missing"})
                continue

            img_b64 = encode_image_base64(abs_img_path)
            group_size = len(picked)

            for idx_in_group, q in enumerate(picked):
                options = (q.get("choices", []) or [])
                scores, _ = onehot_from_answer_letter(q.get("answer",""), options)
                if 1 not in scores:
                    if _unonehot_dumped < _unonehot_cap:
                        with open(os.path.join(output_dir, "answer_unonehot_samples.txt"), "a", encoding="utf-8") as ff:
                            ff.write(json.dumps({
                                "image_rel": rel,
                                "answer": q.get("answer",""),
                                "choices": options,
                                "question": (q.get("question","") or "")[:300]
                            }, ensure_ascii=False) + "\n")
                        _unonehot_dumped += 1
                    key = (os.path.dirname(rel).replace(f"{REL_IMG_PREFIX}/", "", 1), "answer_unonehot")
                    drops_by_subdir_reason[key] += 1
                    if len(drops_examples[key]) < 5:
                        drops_examples[key].append({"rel": rel, "question_id": q.get("question_id"), "stage": "exp", "note": "unonehot"})
                    continue

                parsed = build_parsed_inputs(rel, q.get("question",""), options)
                meta_obj = {
                    "data_id": q.get("data_id"),
                    "question_id": q.get("question_id"),
                    "data_source": q.get("data_source"),
                    "level": q.get("level"),
                    "subpart": q.get("subpart"),
                    "version": q.get("version"),
                    "data_type": q.get("data_type"),
                    "image_rel": rel,
                    # ✅ 新增：稳定分组键（与 questions_one_image_multiq 对齐）
                    "group_key": q.get("data_id")
                }
                if write_group_tags:
                    meta_obj.update({
                        "index_in_group": idx_in_group,
                        "group_size": group_size,
                    })
                sample = {
                    "parsed_inputs": parsed,
                    "multiple_choice_targets": options,
                    "multiple_choice_scores": scores,
                    "image_base64": img_b64,
                    "meta": meta_obj
                }
                g.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # ⭐ 写完这一张图片对应的所有题目后，给该数据集来源 +1
            source_counter[source] += 1

    # 4) 统计 & 日志
    stats_txt = os.path.join(output_dir, "stats.txt")
    with open(stats_txt, "w", encoding="utf-8") as f:
        f.write(f"Images kept: {len(kept_imgs)}\n")
        f.write(f"Question lines (with image): {kept_q_lines}\n")
        f.write(f"Outputs:\n - {qset_path}\n - {os.path.join(output_dir, REL_IMG_PREFIX)}\n - {os.path.join(output_dir, 'experiment_ready(3).jsonl')}\n")
        f.write(f"Extra logs:\n - {drop_missing_detail_log}\n - {drop_missing_by_dir_log}\n - {copy_errors_log}\n - {rewrites_log}\n")
        f.write(f"New logs:\n - {drops_by_subdir_reason_tsv}\n - {drops_examples_tsv}\n")

    if missing:
        with open(missing_log, "w", encoding="utf-8") as f:
            for did, reason in missing:
                f.write(f"{did}\t{reason}\n")

    if 'missing_detail_records' in locals() and missing_detail_records:
        with open(drop_missing_detail_log, "w", encoding="utf-8") as f:
            f.write("data_id\tquestion_id\timage_rel\tfixed_rel\treason\n")
            for data_id, qid, relp, fixed_rel, reason in missing_detail_records:
                f.write(f"{data_id}\t{qid}\t{relp}\t{fixed_rel}\t{reason}\n")

    if 'missing_by_dir_counter' in locals() and missing_by_dir_counter:
        with open(drop_missing_by_dir_log, "w", encoding="utf-8") as f:
            f.write("rel_dir\tcount\n")
            for rel_dir, cnt in missing_by_dir_counter.most_common():
                f.write(f"{rel_dir}\t{cnt}\n")

    # 新增：按“次级目录×原因”的 TSV 输出
    if drops_by_subdir_reason:
        with open(drops_by_subdir_reason_tsv, "w", encoding="utf-8") as f:
            f.write("subdir\treason\tcount\n")
            for (subdir, reason), cnt in sorted(drops_by_subdir_reason.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
                f.write(f"{subdir}\t{reason}\t{cnt}\n")
        with open(drops_examples_tsv, "w", encoding="utf-8") as f:
            f.write("subdir\treason\tjson\n")
            for (subdir, reason), lst in drops_examples.items():
                for ex in lst:
                    f.write(f"{subdir}\t{reason}\t{json.dumps(ex, ensure_ascii=False)}\n")

    print("==== 完成 ====")
    print("问题集：", qset_path)
    print("图片集目录：", os.path.join(output_dir, REL_IMG_PREFIX))
    print("实验集：", os.path.join(output_dir, "experiment_ready(3).jsonl"))
    print("统计：", stats_txt)
    if missing:
        print(f"[Warn] {len(missing)} 个 data_id 未匹配到图片，见 {missing_log}")
    print(f"[Info] 路径重写日志：", rewrites_log)
    print(f"[Info] 缺图详情：", drop_missing_detail_log)
    print(f"[Info] 缺图按目录聚合：", drop_missing_by_dir_log)
    print(f"[Info] 复制失败等错误：", copy_errors_log)
    print(f"[Info] 次级目录×原因汇总：", drops_by_subdir_reason_tsv)
    print(f"[Info] 次级目录示例：", drops_examples_tsv)
    print(f"[Time] 用时 {time.time()-t0:.2f} s")

# ========= CLI =========
def parse_args():
    ap = argparse.ArgumentParser(description="一图多题（v3.5 no-zip：纯目录版，方案B）")
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS_JSON, help="SEED-Bench-H.json")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="输出目录（会创建 images 子目录）")
    ap.add_argument("--min_repeat", type=int, default=2, help="按 data_id 聚合的最少题目数")
    ap.add_argument("--images", action="append", default=None, help="图片来源（目录），可多次传")
    ap.add_argument("--strict_min_q", type=int, default=2, help="实验集阶段：每图最少题目数")
    ap.add_argument("--require_single_image_type", action="store_true",
                    help="仅保留 meta.data_type == 'Single Image'")
    ap.add_argument("--filter_multi_in_qset", action="store_true",
                    help="在问题集阶段也过滤“多图题”")
    ap.add_argument("--take_per_image", type=int, default=3,
                    help="每张图写入题目数；不足则整组丢弃，超过则挑选该数目")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--pick_strategy", choices=["first","random"], default="first",
                    help="当某图题数 > take_per_image 时的挑选策略")
    ap.add_argument("--write_group_tags", action="store_true",
                    help="在 meta 中写入 group_key/index_in_group/group_size")
    ap.add_argument("--clean_out_images", action="store_true",
                    help="运行前清空 out/images 目录")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    images_src = args.images if args.images is not None else DEFAULT_IMAGES_SRC
    run(args.questions, images_src, args.out,
        min_repeat=args.min_repeat,
        strict_min_q=args.strict_min_q,
        require_single_image_type=args.require_single_image_type,
        filter_multi_in_qset=args.filter_multi_in_qset,
        take_per_image=args.take_per_image,
        seed=args.seed,
        pick_strategy=args.pick_strategy,
        write_group_tags=args.write_group_tags,
        clean_out_images=args.clean_out_images)
