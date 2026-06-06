import os
import gc
import re
import csv
import math
import json
import time
import argparse
import inspect
from pathlib import Path
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 基础路径配置
# ============================================================

BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "./baseline_model"
)

# 如果你之前有 ./baseline_model，就用它；没有就自动回退到 BASE_MODEL_PATH
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "./baseline_model")
if not Path(BASE_MODEL_DIR).exists():
    BASE_MODEL_DIR = BASE_MODEL_PATH

DATA_VALID = os.environ.get(
    "DATA_VALID",
    "./data/wikitext-103/wiki.valid.tokens"
)

LAOPPE_STATE_DICT = os.environ.get(
    "LAOPPE_STATE_DICT",
    "./checkpoints/laoppe_l2_th2048_delta.pt"
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "./bclass_plus_results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = os.environ.get("DEVICE", "cuda:0")
DTYPE_NAME = os.environ.get("DTYPE", "bf16").lower()
DTYPE = torch.bfloat16 if DTYPE_NAME in ["bf16", "bfloat16"] else torch.float16

START_LAYER = int(os.environ.get("START_LAYER", "14"))
END_LAYER = int(os.environ.get("END_LAYER", "16"))
THRESHOLD = int(os.environ.get("THRESHOLD", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "512"))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", "0.05"))

SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(SEED)


# ============================================================
# 工具函数
# ============================================================

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = 131072
    return tok


def get_rope_theta_from_config(cfg):
    old_params = getattr(cfg, "rope_parameters", None)
    if old_params is None:
        old_params = {}
    elif not isinstance(old_params, dict):
        old_params = dict(old_params)

    rope_theta = (
        old_params.get("rope_theta", None)
        or old_params.get("base", None)
        or getattr(cfg, "rope_theta", None)
        or 500000.0
    )
    return float(rope_theta)


def apply_rope_scaling_config(cfg, rope_type, factor):
    """
    兼容你当前 transformers 版本的 RoPE 配置写法。

    你的环境之前报过：
    1. KeyError: 'rope_type'
    2. TypeError: NoneType ** Tensor

    所以这里同时写：
    - cfg.rope_parameters["rope_type"]
    - cfg.rope_parameters["rope_theta"]
    - cfg.rope_theta 顶层属性
    - cfg.rope_scaling 兼容旧接口
    """

    rope_theta = get_rope_theta_from_config(cfg)

    old_params = getattr(cfg, "rope_parameters", None)
    if old_params is None:
        old_params = {}
    else:
        old_params = dict(old_params)

    head_dim = (
        old_params.get("head_dim", None)
        or getattr(cfg, "head_dim", None)
        or (cfg.hidden_size // cfg.num_attention_heads)
    )

    partial_rotary_factor = (
        old_params.get("partial_rotary_factor", None)
        or getattr(cfg, "partial_rotary_factor", None)
        or 1.0
    )

    original_max_position_embeddings = (
        old_params.get("original_max_position_embeddings", None)
        or getattr(cfg, "max_position_embeddings", None)
        or 131072
    )

    # 顶层属性必须写，否则你当前环境可能出现 base=None
    cfg.rope_theta = float(rope_theta)
    cfg.head_dim = int(head_dim)
    cfg.partial_rotary_factor = float(partial_rotary_factor)

    # YaRN / dynamic / linear 通用字段
    new_params = dict(old_params)
    new_params.update({
        "rope_type": rope_type,
        "factor": float(factor),
        "rope_theta": float(rope_theta),
        "base": float(rope_theta),
        "head_dim": int(head_dim),
        "partial_rotary_factor": float(partial_rotary_factor),
        "original_max_position_embeddings": int(original_max_position_embeddings),
    })

    # YaRN 专用字段：不写也有默认，但显式写方便复现
    if rope_type == "yarn":
        new_params.setdefault("attention_factor", 0.1 * math.log(float(factor)) + 1.0)
        new_params.setdefault("beta_fast", 32.0)
        new_params.setdefault("beta_slow", 1.0)

    cfg.rope_parameters = new_params

    # 旧接口兼容
    cfg.rope_scaling = {
        "type": rope_type,
        "rope_type": rope_type,
        "factor": float(factor),
        "original_max_position_embeddings": int(original_max_position_embeddings),
    }

    # 让配置最大长度至少覆盖目标
    try:
        cfg.max_position_embeddings = max(
            int(getattr(cfg, "max_position_embeddings", 0)),
            int(original_max_position_embeddings * float(factor))
        )
    except Exception:
        pass

    print(
        f"✅ RoPE scaling: rope_type={rope_type}, factor={factor}, "
        f"rope_theta={rope_theta}, original_max={original_max_position_embeddings}"
    )

    return cfg


def load_base_model(path, rope_type=None, factor=None):
    cfg = AutoConfig.from_pretrained(path)

    if rope_type is not None:
        cfg = apply_rope_scaling_config(cfg, rope_type=rope_type, factor=factor)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=DTYPE,
        config=cfg,
        device_map={"": DEVICE},
    )
    model.eval()
    return model


def load_laoppe_model(path):
    """
    加载你的 LAOPPE_L2 模型。
    依赖已有文件：llama_rope_laoppe_score_bias.py
    """
    from llama_rope_laoppe_score_bias import replace_llama_attention_score_bias

    model = load_base_model(path)

    # 兼容不同函数签名
    kwargs = {
        "start_layer": START_LAYER,
        "end_layer": END_LAYER,
        "threshold": THRESHOLD,
        "temperature": TEMPERATURE,
        "alpha_max": ALPHA_MAX,
    }

    sig = inspect.signature(replace_llama_attention_score_bias)
    used_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    model = replace_llama_attention_score_bias(model, **used_kwargs)

    if not Path(LAOPPE_STATE_DICT).exists():
        raise FileNotFoundError(f"LAOPPE_STATE_DICT 不存在: {LAOPPE_STATE_DICT}")

    sd = torch.load(LAOPPE_STATE_DICT, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"✅ LAOPPE loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def build_model(method):
    method = method.strip()

    if method == "RoPE":
        return load_base_model(BASE_MODEL_DIR)

    if method == "YaRN_x2":
        return load_base_model(BASE_MODEL_DIR, rope_type="yarn", factor=2.0)

    if method == "YaRN_x4":
        return load_base_model(BASE_MODEL_DIR, rope_type="yarn", factor=4.0)

    if method == "NTK_dynamic_x2":
        return load_base_model(BASE_MODEL_DIR, rope_type="dynamic", factor=2.0)

    if method == "PI_linear_x2":
        return load_base_model(BASE_MODEL_DIR, rope_type="linear", factor=2.0)

    if method == "LAOPPE_L2":
        return load_laoppe_model(BASE_MODEL_PATH)

    raise ValueError(f"Unknown method: {method}")


def parse_methods(s):
    return [x.strip() for x in s.split(",") if x.strip()]


# ============================================================
# WikiText 读取
# ============================================================

def read_valid_text():
    path = Path(DATA_VALID)
    if not path.exists():
        raise FileNotFoundError(f"DATA_VALID 不存在: {path}")

    txt = path.read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt


def tokenize_valid(tokenizer):
    txt = read_valid_text()
    enc = tokenizer(txt, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0]
    print(f"valid token 总数: {ids.numel()}")
    return ids


def make_fixed_samples(token_ids, seq_len, num_samples):
    """
    固定采样：每个方法都用相同样本，保证公平。
    """
    total = token_ids.numel()
    if total < seq_len + 1:
        raise ValueError(f"valid tokens 不足，total={total}, seq_len={seq_len}")

    # 均匀采样，避免随机差异
    max_start = total - seq_len - 1
    if num_samples == 1:
        starts = [0]
    else:
        starts = [
            int(i * max_start / (num_samples - 1))
            for i in range(num_samples)
        ]

    samples = []
    for st in starts:
        samples.append(token_ids[st: st + seq_len].clone())
    return samples


# ============================================================
# 8192 PPL
# ============================================================

@torch.no_grad()
def calc_full_ppl(model, input_ids):
    x = input_ids.unsqueeze(0).to(DEVICE)
    attn = torch.ones_like(x, device=DEVICE)

    out = model(
        input_ids=x,
        attention_mask=attn,
        labels=x,
        use_cache=False,
    )
    loss = out.loss.float().item()
    ppl = math.exp(loss) if loss < 20 else float("inf")
    return ppl


@torch.no_grad()
def calc_sliding_ppl(model, input_ids, window_size=4096, stride=2048):
    """
    滑窗 PPL。
    注意：这是 OOM fallback。
    位置会在每个窗口内重新计算，因此它不是严格 full-context 8192 PPL。
    论文中应写作 sliding-window PPL。
    """
    ids = input_ids.to(DEVICE)
    seq_len = ids.numel()

    nll_sum = 0.0
    token_count = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + window_size, seq_len)
        if end <= begin + 1:
            break

        trg_len = end - prev_end
        if trg_len <= 0:
            continue

        input_chunk = ids[begin:end].unsqueeze(0)
        labels = input_chunk.clone()

        # 只计算当前新增部分 token 的 loss
        if labels.size(1) > trg_len:
            labels[:, :-trg_len] = -100

        attn = torch.ones_like(input_chunk, device=DEVICE)

        out = model(
            input_ids=input_chunk,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )

        nll_sum += out.loss.float().item() * trg_len
        token_count += trg_len

        prev_end = end
        if end >= seq_len:
            break

    if token_count == 0:
        return float("nan")

    return math.exp(nll_sum / token_count)


def run_ppl8192(args):
    tokenizer = load_tokenizer(BASE_MODEL_PATH)
    token_ids = tokenize_valid(tokenizer)
    samples = make_fixed_samples(
        token_ids,
        seq_len=args.seq_len,
        num_samples=args.num_samples
    )

    out_sample = OUT_DIR / "ppl8192_sample_level.csv"
    out_summary = OUT_DIR / "ppl8192_results.csv"

    methods = parse_methods(args.methods)

    sample_rows = []

    for method in methods:
        cleanup()
        print("=" * 80)
        print(f"加载模型: {method}")
        print("=" * 80)

        try:
            model = build_model(method)
        except Exception as e:
            print(f"❌ {method} 加载失败: {repr(e)}")
            continue

        for sid, ids in enumerate(samples):
            mode = "full"
            try:
                ppl = calc_full_ppl(model, ids)
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ {method} sample={sid} full {args.seq_len} OOM，改用滑窗")
                cleanup()
                mode = f"sliding_w{args.window_size}_s{args.stride}"
                ppl = calc_sliding_ppl(
                    model,
                    ids,
                    window_size=args.window_size,
                    stride=args.stride
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ {method} sample={sid} full {args.seq_len} OOM，改用滑窗")
                    cleanup()
                    mode = f"sliding_w{args.window_size}_s{args.stride}"
                    ppl = calc_sliding_ppl(
                        model,
                        ids,
                        window_size=args.window_size,
                        stride=args.stride
                    )
                else:
                    raise

            print(f"{method} seq_len={args.seq_len} sample={sid} ppl={ppl:.4f} mode={mode}")

            sample_rows.append({
                "method": method,
                "seq_len": args.seq_len,
                "sample_id": sid,
                "ppl": ppl,
                "eval_mode": mode,
            })

        del model
        cleanup()

    with out_sample.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "seq_len", "sample_id", "ppl", "eval_mode"]
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    # 汇总
    groups = defaultdict(list)
    modes = defaultdict(set)

    for r in sample_rows:
        if math.isfinite(float(r["ppl"])):
            groups[r["method"]].append(float(r["ppl"]))
            modes[r["method"]].add(r["eval_mode"])

    summary_rows = []

    for method, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        summary_rows.append({
            "method": method,
            "seq_len": args.seq_len,
            "count": n,
            "mean_ppl": mean,
            "std_ppl": std,
            "min_ppl": min(vals),
            "max_ppl": max(vals),
            "eval_modes": "|".join(sorted(modes[method])),
        })

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "seq_len", "count",
                "mean_ppl", "std_ppl", "min_ppl", "max_ppl",
                "eval_modes",
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✅ 保存 sample-level: {out_sample}")
    print(f"✅ 保存 summary: {out_summary}")


# ============================================================
# Needle with YaRN
# ============================================================

def make_needle_prompt(seq_len, sample_id, tokenizer):
    """
    构造与你之前类似的 Needle 样本。
    """
    key = f"HMN{sample_id:04d}"
    needle = f"The secret passcode is {key}."

    question = "\nQuestion: What is the secret passcode?\nAnswer:"

    filler = (
        "This is ordinary background text about history, science, and literature. "
        "It does not contain useful information for the question. "
    )

    # 估算需要多少 filler
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]

    target_context_tokens = seq_len - len(needle_ids) - len(question_ids) - 8
    repeat = max(1, target_context_tokens // max(1, len(filler_ids)))

    # 将 needle 放在靠前、中间、靠后不同位置
    pos_ratio = (sample_id % 5) / 4.0
    before_repeat = int(repeat * pos_ratio)
    after_repeat = repeat - before_repeat

    text = (filler * before_repeat) + "\n" + needle + "\n" + (filler * after_repeat) + question

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    # 截断或补齐到 seq_len 附近
    if len(ids) > seq_len:
        ids = ids[-seq_len:]

    return ids, key


@torch.no_grad()
def generate_answer(model, tokenizer, input_ids, max_new_tokens=32):
    x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    attn = torch.ones_like(x, device=DEVICE)

    out = model.generate(
        input_ids=x,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    gen = out[0, x.size(1):]
    return tokenizer.decode(gen, skip_special_tokens=True)


def run_yarn_needle(args):
    tokenizer = load_tokenizer(BASE_MODEL_PATH)
    methods = parse_methods(args.methods)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    out_path = OUT_DIR / "needle_yarn_results.csv"
    out_sample = OUT_DIR / "needle_yarn_sample_level.csv"

    sample_rows = []
    summary_rows = []

    for method in methods:
        cleanup()
        print("=" * 80)
        print(f"加载模型: {method}")
        print("=" * 80)

        try:
            model = build_model(method)
        except Exception as e:
            print(f"❌ {method} 加载失败: {repr(e)}")
            continue

        for seq_len in seq_lens:
            success = 0
            total = args.num_samples

            for sid in range(args.num_samples):
                ids, key = make_needle_prompt(seq_len, sid, tokenizer)
                pred = generate_answer(
                    model,
                    tokenizer,
                    ids,
                    max_new_tokens=args.max_new_tokens
                )

                hit = key in pred
                success += int(hit)

                print(
                    f"{method} {seq_len} {sid} "
                    f"{'hit' if hit else 'miss'} {key} | {pred[:80].replace(chr(10), ' ')}"
                )

                sample_rows.append({
                    "method": method,
                    "seq_len": seq_len,
                    "sample_id": sid,
                    "key": key,
                    "hit": int(hit),
                    "prediction": pred.replace("\n", "\\n")[:300],
                })

            acc = success / total
            summary_rows.append({
                "method": method,
                "seq_len": seq_len,
                "success": success,
                "total": total,
                "accuracy": acc,
            })

        del model
        cleanup()

    with out_sample.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "seq_len", "sample_id", "key", "hit", "prediction"]
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "seq_len", "success", "total", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✅ 保存 Needle sample-level: {out_sample}")
    print(f"✅ 保存 Needle summary: {out_path}")


# ============================================================
# LongBench 小子集
# ============================================================

def load_jsonl_dataset(path, max_samples):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def try_load_longbench(task, max_samples):
    """
    离线版本：只从本地读取 LongBench jsonl，不再联网。
    需要文件放在：
    ./LongBench/{task}.jsonl
    或 ./LongBench/data/{task}.jsonl
    或 ./LongBench/{task}/test.jsonl
    """
    local_dir = Path(os.environ.get("LONGBENCH_DIR", "./LongBench"))

    candidates = [
        local_dir / f"{task}.jsonl",
        local_dir / "data" / f"{task}.jsonl",
        local_dir / task / "test.jsonl",
        local_dir / task / f"{task}.jsonl",
    ]

    print(f"查找本地 LongBench task={task}")
    for p in candidates:
        print("  candidate:", p)
        if p.exists():
            rows = load_jsonl_dataset(p, max_samples)
            print(f"✅ 本地 LongBench 加载成功: {p}, 样本数={len(rows)}")
            return rows, f"local:{p}"

    raise FileNotFoundError(
        f"没有找到 {task}.jsonl。请把文件放到 ./LongBench/ 或 ./LongBench/data/ 下。"
    )


def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", s)
    return s.strip()


def token_f1(pred, ans):
    pred_toks = normalize_text(pred).split()
    ans_toks = normalize_text(ans).split()

    if len(pred_toks) == 0 or len(ans_toks) == 0:
        return 0.0

    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1

    overlap = 0
    for t in ans_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(ans_toks)
    return 2 * precision * recall / (precision + recall)


def score_prediction(pred, answers):
    if not isinstance(answers, list):
        answers = [answers]

    pred_norm = normalize_text(pred)

    em = 0
    f1 = 0.0

    for ans in answers:
        ans_norm = normalize_text(ans)

        # LongBench 里一些 retrieval 任务答案很短，包含即可
        if ans_norm and (ans_norm in pred_norm or pred_norm in ans_norm):
            em = 1

        f1 = max(f1, token_f1(pred, ans))

    return em, f1


def build_longbench_prompt(task, sample):
    context = sample.get("context", "")
    question = sample.get("input", "")

    # 尽量短，避免 base model 生成太散
    if "passage_retrieval" in task:
        prompt = (
            "Read the following passages and answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Only output the target passage ID or the shortest answer.\n"
            "Answer:"
        )
    elif task in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]:
        prompt = (
            "Read the following document and answer the question briefly.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    else:
        prompt = (
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return prompt


@torch.no_grad()
def generate_longbench(model, tokenizer, prompt, max_input_tokens, max_new_tokens):
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # 保留后半部分，避免超长 OOM；LongBench 真实任务记录实际输入长度
    if len(ids) > max_input_tokens:
        ids = ids[-max_input_tokens:]

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    attn = torch.ones_like(x, device=DEVICE)

    out = model.generate(
        input_ids=x,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    gen = out[0, x.size(1):]
    pred = tokenizer.decode(gen, skip_special_tokens=True)

    return pred, len(ids)


def run_longbench(args):
    tokenizer = load_tokenizer(BASE_MODEL_PATH)
    methods = parse_methods(args.methods)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    out_sample = OUT_DIR / "longbench_subset_sample_level.csv"
    out_summary = OUT_DIR / "longbench_subset_results.csv"

    all_sample_rows = []
    summary_rows = []

    # 预先加载数据，保证各方法同一批样本
    task_data = {}
    for task in tasks:
        ds, ds_name = try_load_longbench(task, args.max_samples)
        task_data[task] = (ds, ds_name)

    for method in methods:
        cleanup()
        print("=" * 80)
        print(f"加载模型: {method}")
        print("=" * 80)

        try:
            model = build_model(method)
        except Exception as e:
            print(f"❌ {method} 加载失败: {repr(e)}")
            continue

        for task in tasks:
            ds, ds_name = task_data[task]
            em_list = []
            f1_list = []
            lengths = []

            for i, sample in enumerate(ds):
                prompt = build_longbench_prompt(task, sample)
                answers = sample.get("answers", [])

                try:
                    pred, input_tokens = generate_longbench(
                        model,
                        tokenizer,
                        prompt,
                        max_input_tokens=args.max_input_tokens,
                        max_new_tokens=args.max_new_tokens,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"⚠️ OOM: {method} {task} sample={i}，跳过")
                    cleanup()
                    continue
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠️ OOM: {method} {task} sample={i}，跳过")
                        cleanup()
                        continue
                    else:
                        raise

                em, f1 = score_prediction(pred, answers)
                em_list.append(em)
                f1_list.append(f1)
                lengths.append(input_tokens)

                print(
                    f"{method} {task} sample={i} "
                    f"em={em} f1={f1:.4f} input_tokens={input_tokens} "
                    f"pred={pred[:80].replace(chr(10), ' ')}"
                )

                all_sample_rows.append({
                    "method": method,
                    "task": task,
                    "sample_id": i,
                    "input_tokens": input_tokens,
                    "em": em,
                    "f1": f1,
                    "answers": json.dumps(answers, ensure_ascii=False),
                    "prediction": pred.replace("\n", "\\n")[:500],
                })

            n = len(em_list)
            if n > 0:
                summary_rows.append({
                    "method": method,
                    "task": task,
                    "dataset": ds_name,
                    "count": n,
                    "mean_input_tokens": sum(lengths) / n,
                    "em": sum(em_list) / n,
                    "f1": sum(f1_list) / n,
                })

        del model
        cleanup()

    with out_sample.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "task", "sample_id", "input_tokens",
                "em", "f1", "answers", "prediction"
            ]
        )
        writer.writeheader()
        writer.writerows(all_sample_rows)

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "task", "dataset", "count",
                "mean_input_tokens", "em", "f1"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✅ 保存 LongBench sample-level: {out_sample}")
    print(f"✅ 保存 LongBench summary: {out_summary}")


# ============================================================
# 汇总 Markdown
# ============================================================

def csv_to_md(path):
    path = Path(path)
    if not path.exists():
        return f"\n文件不存在: {path}\n"

    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return f"\n文件为空: {path}\n"

    headers = rows[0].keys()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        vals = []
        for h in headers:
            v = r[h]
            try:
                fv = float(v)
                if abs(fv) < 100000:
                    v = f"{fv:.4f}"
            except Exception:
                pass
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def run_summary(args):
    out = OUT_DIR / "plus_experiment_summary.md"

    with out.open("w", encoding="utf-8") as f:
        f.write("# B 类期刊进一步补充实验结果\n\n")

        f.write("## 1. 8192 PPL / 滑窗 PPL\n\n")
        f.write(csv_to_md(OUT_DIR / "ppl8192_results.csv"))
        f.write("\n\n")

        f.write("## 2. YaRN Needle 对比\n\n")
        f.write(csv_to_md(OUT_DIR / "needle_yarn_results.csv"))
        f.write("\n\n")

        f.write("## 3. LongBench 小子集\n\n")
        f.write(csv_to_md(OUT_DIR / "longbench_subset_results.csv"))
        f.write("\n\n")

        f.write("## 4. 写作建议\n\n")
        f.write(
            "若 YaRN_x2 在 PPL 或 Needle 上优于 LAOPPE_L2，应在论文中承认 YaRN 是更强 RoPE 扩展基线；"
            "若 LAOPPE_L2 在 LongBench 某些任务上更稳，可强调本文方法在冻结主模型、极少新增参数条件下的性价比。"
            "8192 若使用 sliding-window PPL，必须在表格中标注 eval_mode，不能与 full-context PPL 混写。\n"
        )

    print(f"✅ 保存汇总: {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("ppl8192")
    p1.add_argument("--methods", type=str, default="RoPE,YaRN_x2,LAOPPE_L2")
    p1.add_argument("--seq_len", type=int, default=8192)
    p1.add_argument("--num_samples", type=int, default=50)
    p1.add_argument("--window_size", type=int, default=4096)
    p1.add_argument("--stride", type=int, default=2048)

    p2 = sub.add_parser("needle_yarn")
    p2.add_argument("--methods", type=str, default="RoPE,YaRN_x2,NTK_dynamic_x2,LAOPPE_L2")
    p2.add_argument("--seq_lens", type=str, default="2048,4096,8192")
    p2.add_argument("--num_samples", type=int, default=20)
    p2.add_argument("--max_new_tokens", type=int, default=32)

    p3 = sub.add_parser("longbench")
    p3.add_argument("--methods", type=str, default="RoPE,YaRN_x2,LAOPPE_L2")
    p3.add_argument("--tasks", type=str, default="passage_retrieval_en,qasper")
    p3.add_argument("--max_samples", type=int, default=30)
    p3.add_argument("--max_input_tokens", type=int, default=8192)
    p3.add_argument("--max_new_tokens", type=int, default=64)

    p4 = sub.add_parser("summary")

    args = parser.parse_args()

    if args.mode == "ppl8192":
        run_ppl8192(args)
    elif args.mode == "needle_yarn":
        run_yarn_needle(args)
    elif args.mode == "longbench":
        run_longbench(args)
    elif args.mode == "summary":
        run_summary(args)


if __name__ == "__main__":
    main()
