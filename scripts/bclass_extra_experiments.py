import os, gc, csv, json, math, time, random, argparse, inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

from llama_rope_laoppe_score_bias import replace_llama_attention_score_bias


BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "./baseline_model")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "./baseline_model")
DATA_TRAIN = os.environ.get("DATA_TRAIN", "./data/wikitext-103/wiki.train.tokens")
DATA_VALID = os.environ.get("DATA_VALID", "./data/wikitext-103/wiki.valid.tokens")

DEVICE = os.environ.get("DEVICE", "cuda:0")
DTYPE = torch.bfloat16

LAOPPE_STATE_DICT = os.environ.get("LAOPPE_STATE_DICT", "./checkpoints/laoppe_l2_th2048_delta.pt")
ALIBI_SAVE_DIR = os.environ.get("ALIBI_SAVE_DIR", "./bclass_results/alibi_L2")
ALIBI_STATE_DICT = os.environ.get("ALIBI_STATE_DICT", "./bclass_results/alibi_L2/full_state_dict.pt")

START_LAYER = int(os.environ.get("START_LAYER", "14"))
END_LAYER = int(os.environ.get("END_LAYER", "16"))
THRESHOLD = int(os.environ.get("THRESHOLD", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "512"))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", "0.05"))

BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "4096"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))
LR = float(os.environ.get("LEARNING_RATE", "1e-4"))

TEST_LENGTHS = [int(x) for x in os.environ.get("TEST_LENGTHS", "512,1024,2048,4096").split(",")]
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))
SEED = int(os.environ.get("SEED", "42"))


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_tokenizer(path=BASE_MODEL_DIR):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    tok.pad_token = tok.eos_token
    return tok


def load_model(path, rope_scaling=None):
    cfg = AutoConfig.from_pretrained(path)

    if rope_scaling is not None:
        rs = dict(rope_scaling)

        rope_type = rs.get("rope_type", rs.get("type", None))
        factor = float(rs.get("factor", 1.0))

        if rope_type is None:
            raise ValueError(f"Invalid rope_scaling: {rope_scaling}")

        old_params = getattr(cfg, "rope_parameters", None)
        if old_params is None:
            old_params = {}
        else:
            old_params = dict(old_params)

        rope_theta = (
            old_params.get("rope_theta", None)
            or old_params.get("base", None)
            or getattr(cfg, "rope_theta", None)
            or 500000.0
        )

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

        # 关键修复：当前 transformers 的 modeling_rope_utils 会读顶层 cfg.rope_theta
        cfg.rope_theta = float(rope_theta)
        cfg.head_dim = int(head_dim)
        cfg.partial_rotary_factor = float(partial_rotary_factor)

        new_params = dict(old_params)
        new_params.update({
            "rope_type": rope_type,
            "factor": factor,
            "rope_theta": float(rope_theta),
            "base": float(rope_theta),
            "head_dim": int(head_dim),
            "partial_rotary_factor": float(partial_rotary_factor),
        })

        if "original_max_position_embeddings" not in new_params:
            new_params["original_max_position_embeddings"] = getattr(
                cfg, "max_position_embeddings", 131072
            )

        cfg.rope_parameters = new_params

        cfg.rope_scaling = {
            "type": rope_type,
            "rope_type": rope_type,
            "factor": factor,
        }

        print(
            f"✅ RoPE scaling enabled: "
            f"rope_type={rope_type}, factor={factor}, rope_theta={cfg.rope_theta}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=DTYPE,
        config=cfg,
        device_map={"": DEVICE},
    )
    model.eval()
    return model


def load_valid_tokens(tokenizer):
    raw = load_dataset("text", data_files={"valid": DATA_VALID})
    sep = tokenizer.encode("\n\n", add_special_tokens=False)
    ids_all = []
    for t in raw["valid"]["text"]:
        t = t.strip()
        if not t:
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            ids_all.extend(ids)
            ids_all.extend(sep)
    print(f"valid 总 token 数: {len(ids_all)}")
    return ids_all


def make_positions(total, max_len, n, seed):
    rng = random.Random(seed)
    max_start = total - max_len - 1
    return rng.sample(range(max_start + 1), n)


def calc_ppl(model, token_ids):
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    ppl = math.exp(out.loss.item())
    del input_ids, out
    return ppl


def eval_ppl_model(name, model, token_ids, starts, output_rows):
    print("=" * 80)
    print(f"评估: {name}")
    print("=" * 80)

    for L in TEST_LENGTHS:
        vals = []
        for idx, s in enumerate(starts):
            seg = token_ids[s:s+L]
            try:
                vals.append(calc_ppl(model, seg))
            except RuntimeError as e:
                print(f"{name} 长度 {L} 样本 {idx} 失败: {str(e)[:100]}")
                clear_cuda()

        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((x - mean) ** 2 for x in vals) / max(1, len(vals)-1)) ** 0.5
        else:
            mean, std = None, None

        print(f"{name:>16} | len={L:>5} | n={len(vals):>3} | ppl={mean} | std={std}")

        for i, v in enumerate(vals):
            output_rows.append({
                "method": name,
                "seq_len": L,
                "sample_id": i,
                "ppl": v
            })


# =========================
# ALiBi Score Bias
# =========================

def get_alibi_slopes(n):
    def slopes_power_2(m):
        start = 2 ** (-(2 ** -(math.log2(m) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(m)]
    if math.log2(n).is_integer():
        return torch.tensor(slopes_power_2(n), dtype=torch.float32)
    p = 2 ** math.floor(math.log2(n))
    return torch.tensor(slopes_power_2(p) + get_alibi_slopes(2*p).tolist()[0::2][:n-p], dtype=torch.float32)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h*n_rep, s, d)


class ALiBiBias(nn.Module):
    def __init__(self, n_heads, alpha_max=0.05):
        super().__init__()
        self.register_buffer("slopes", get_alibi_slopes(n_heads), persistent=False)
        self.alpha_raw = nn.Parameter(torch.zeros(n_heads))
        self.alpha_max = alpha_max

    def forward(self, q_len, kv_len, device, dtype):
        q = torch.arange(q_len, device=device).view(q_len, 1)
        k = torch.arange(kv_len, device=device).view(1, kv_len)
        dist = torch.clamp(q - k, min=0).float()
        slopes = self.slopes.to(device).view(1, -1, 1, 1)
        alpha = self.alpha_max * torch.tanh(self.alpha_raw).view(1, -1, 1, 1).to(device)
        return torch.clamp(-alpha * slopes * dist.view(1, 1, q_len, kv_len), -2.0, 2.0).to(dtype)


class LlamaAttentionALiBi(nn.Module):
    def __init__(self, old_attn, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(old_attn, "head_dim", self.hidden_size // self.num_heads)
        self.q_proj = old_attn.q_proj
        self.k_proj = old_attn.k_proj
        self.v_proj = old_attn.v_proj
        self.o_proj = old_attn.o_proj
        self.alibi_bias = ALiBiBias(self.num_heads, ALPHA_MAX)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            try:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            except TypeError:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        kv_len = k.shape[-2]
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = attn + self.alibi_bias(q_len, kv_len, attn.device, attn.dtype)

        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, :kv_len]

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)

        if not output_attentions:
            attn = None
        return out, attn


def replace_alibi(model):
    for i in range(START_LAYER, END_LAYER):
        model.model.layers[i].self_attn = LlamaAttentionALiBi(
            model.model.layers[i].self_attn,
            model.config,
            i
        )
        print(f"✅ ALiBi 接入: layer {i}")
    return model


def tokenize_fn(examples, tokenizer):
    return tokenizer(examples["text"], add_special_tokens=False)


def group_texts(examples):
    ids = []
    for x in examples["input_ids"]:
        ids.extend(x)

    total = len(ids) // BLOCK_SIZE * BLOCK_SIZE
    ids = ids[:total]

    chunks = [
        ids[i:i + BLOCK_SIZE]
        for i in range(0, total, BLOCK_SIZE)
    ]

    return {
        "input_ids": chunks,
        "attention_mask": [[1] * BLOCK_SIZE for _ in chunks],
        "labels": [x.copy() for x in chunks],
    }


def train_alibi():
    tokenizer = load_tokenizer(BASE_MODEL_PATH)
    raw = load_dataset("text", data_files={"train": DATA_TRAIN})
    ds = raw["train"].filter(lambda x: len(x["text"].strip()) > 50)
    ds = ds.shuffle(seed=42).select(range(min(40000, len(ds))))
    ds = ds.map(tokenize_fn, batched=True, num_proc=12, fn_kwargs={"tokenizer": tokenizer}, remove_columns=["text"])
    ds = ds.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=12,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
        desc=f"Group texts into block_size={BLOCK_SIZE}",
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) == BLOCK_SIZE)

    model = load_model(BASE_MODEL_PATH)
    model = replace_alibi(model)

    for _, p in model.named_parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "alibi_bias" in n:
            p.requires_grad = True
            print("TRAINABLE:", n, p.numel())

    args_dict = dict(
        output_dir=ALIBI_SAVE_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        bf16=True,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        warmup_steps=30,
        logging_steps=50,
        logging_strategy="steps",
        save_steps=1000,
        save_strategy="steps",
        save_total_limit=1,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=8,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    args_dict = {k:v for k,v in args_dict.items() if k in sig.parameters}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_dict),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer
    )
    trainer.train()

    Path(ALIBI_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ALIBI_SAVE_DIR)
    tokenizer.save_pretrained(ALIBI_SAVE_DIR)
    torch.save(model.state_dict(), Path(ALIBI_SAVE_DIR) / "full_state_dict.pt")
    print("✅ ALiBi 保存完成:", ALIBI_SAVE_DIR)


def load_alibi_trained():
    model = load_model(BASE_MODEL_DIR)
    model = replace_alibi(model)
    sd = torch.load(ALIBI_STATE_DICT, map_location="cpu")
    info = model.load_state_dict(sd, strict=False)
    print("ALiBi missing:", len(info.missing_keys), "unexpected:", len(info.unexpected_keys))
    model.eval()
    return model


def load_laoppe():
    model = load_model(BASE_MODEL_DIR)
    model = replace_llama_attention_score_bias(
        model,
        start_layer=START_LAYER,
        end_layer=END_LAYER,
        threshold=THRESHOLD,
        temperature=TEMPERATURE,
        alpha_max=ALPHA_MAX,
    )
    sd = torch.load(LAOPPE_STATE_DICT, map_location="cpu")
    info = model.load_state_dict(sd, strict=False)
    print("LAOPPE missing:", len(info.missing_keys), "unexpected:", len(info.unexpected_keys))
    model.eval()
    return model


def eval_ppl_all():
    Path("bclass_results").mkdir(exist_ok=True)
    tokenizer = load_tokenizer(BASE_MODEL_DIR)
    ids = load_valid_tokens(tokenizer)
    starts = make_positions(len(ids), max(TEST_LENGTHS), NUM_SAMPLES, SEED)
    rows = []

    experiments = []

    experiments.append(("RoPE", lambda: load_model(BASE_MODEL_DIR)))
    experiments.append(("PI_linear_x2", lambda: load_model(BASE_MODEL_DIR, rope_scaling={"type": "linear", "factor": 2.0})))
    experiments.append(("NTK_dynamic_x2", lambda: load_model(BASE_MODEL_DIR, rope_scaling={"type": "dynamic", "factor": 2.0})))

    if Path(ALIBI_STATE_DICT).exists():
        experiments.append(("ALiBi_trained", load_alibi_trained))
    else:
        print("⚠️ 未找到 ALiBi 权重，跳过 ALiBi_trained:", ALIBI_STATE_DICT)

    if Path(LAOPPE_STATE_DICT).exists():
        experiments.append(("LAOPPE_L2", load_laoppe))

    for name, fn in experiments:
        try:
            model = fn()
            eval_ppl_model(name, model, ids, starts, rows)
            del model
            clear_cuda()
        except Exception as e:
            print(f"❌ {name} 加载或评估失败:", e)
            clear_cuda()

    out = "bclass_results/ppl_sample_level.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seq_len", "sample_id", "ppl"])
        writer.writeheader()
        writer.writerows(rows)

    print("✅ 样本级 PPL 保存:", out)


def needle_test():
    Path("bclass_results").mkdir(exist_ok=True)
    tokenizer = load_tokenizer(BASE_MODEL_DIR)

    methods = {
        "RoPE": lambda: load_model(BASE_MODEL_DIR),
        "PI_linear_x2": lambda: load_model(BASE_MODEL_DIR, rope_scaling={"type": "linear", "factor": 2.0}),
        "NTK_dynamic_x2": lambda: load_model(BASE_MODEL_DIR, rope_scaling={"type": "dynamic", "factor": 2.0}),
    }
    if Path(ALIBI_STATE_DICT).exists():
        methods["ALiBi_trained"] = load_alibi_trained
    if Path(LAOPPE_STATE_DICT).exists():
        methods["LAOPPE_L2"] = load_laoppe

    lengths = [int(x) for x in os.environ.get("NEEDLE_LENGTHS", "2048,4096,8192").split(",")]
    n_cases = int(os.environ.get("NEEDLE_CASES", "20"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "16"))

    filler = "This is ordinary background text about history, science, and literature. "
    rows = []

    for name, fn in methods.items():
        model = fn()
        for L in lengths:
            success = 0
            total = 0
            for i in range(n_cases):
                code = f"HMN{i:04d}"
                needle = f"The secret passcode is {code}. "
                base_text = filler * 20000
                ids = tokenizer.encode(base_text, add_special_tokens=False)
                ids = ids[:max(10, L - 80)]
                insert_pos = random.Random(SEED + i).randint(0, max(1, len(ids)-1))
                needle_ids = tokenizer.encode(needle, add_special_tokens=False)
                ids = ids[:insert_pos] + needle_ids + ids[insert_pos:]
                ids = ids[:L]

                prompt = tokenizer.decode(ids, skip_special_tokens=True)
                prompt += "\nQuestion: What is the secret passcode? Answer:"
                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=L).input_ids.to(DEVICE)

                try:
                    with torch.no_grad():
                        out = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                        )
                    gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                    hit = code in gen
                    success += int(hit)
                    total += 1
                    print(name, L, i, "hit" if hit else "miss", gen[:80])
                except RuntimeError as e:
                    print(name, L, i, "OOM/ERR", str(e)[:80])
                    clear_cuda()

            acc = success / total if total else 0
            rows.append({"method": name, "seq_len": L, "success": success, "total": total, "accuracy": acc})
        del model
        clear_cuda()

    with open("bclass_results/needle_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seq_len", "success", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Needle 结果保存: bclass_results/needle_results.csv")


def paired_ttest():
    import pandas as pd
    from scipy import stats

    path = os.environ.get("PPL_SAMPLE_CSV", "bclass_results/ppl_sample_level.csv")
    df = pd.read_csv(path)

    base = os.environ.get("BASE_METHOD", "RoPE")
    targets = os.environ.get("TARGET_METHODS", "ALiBi_trained,PI_linear_x2,NTK_dynamic_x2,LAOPPE_L2").split(",")

    rows = []
    for L in sorted(df.seq_len.unique()):
        b = df[(df.method == base) & (df.seq_len == L)].sort_values("sample_id")
        for t in targets:
            x = df[(df.method == t) & (df.seq_len == L)].sort_values("sample_id")
            n = min(len(b), len(x))
            if n < 2:
                continue
            bv = b.ppl.values[:n]
            tv = x.ppl.values[:n]
            stat, p = stats.ttest_rel(bv, tv)
            diff = tv - bv
            rows.append({
                "seq_len": L,
                "compare": f"{t} vs {base}",
                "n": n,
                "base_mean": float(bv.mean()),
                "target_mean": float(tv.mean()),
                "mean_diff_target_minus_base": float(diff.mean()),
                "t_stat": float(stat),
                "p_value": float(p),
            })

    out = "bclass_results/paired_ttest_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("✅ paired t-test 保存:", out)
    for r in rows:
        print(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train_alibi", "eval_ppl", "needle", "ttest"])
    args = parser.parse_args()

    if args.mode == "train_alibi":
        train_alibi()
    elif args.mode == "eval_ppl":
        eval_ppl_all()
    elif args.mode == "needle":
        needle_test()
    elif args.mode == "ttest":
        paired_ttest()


if __name__ == "__main__":
    main()
