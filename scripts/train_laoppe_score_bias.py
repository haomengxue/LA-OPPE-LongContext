import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gc
import inspect
import multiprocessing

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
    set_seed,
)

from llama_rope_laoppe_score_bias import (
    replace_llama_attention_score_bias,
    collect_laoppe_metrics,
)

# ============================================================
# 配置区
# ============================================================

BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "./baseline_model",
)

LOAD_STATE_DICT = os.environ.get(
    "LOAD_STATE_DICT",
    "",
)

SAVE_DIR = os.environ.get(
    "SAVE_DIR",
    "./hybrid_model_score_bias_4096",
)

TRAIN_DATA_PATH = os.environ.get(
    "TRAIN_DATA_PATH",
    "./data/wikitext-103/wiki.train.tokens",
)

VALID_DATA_PATH = os.environ.get(
    "VALID_DATA_PATH",
    "./data/wikitext-103/wiki.valid.tokens",
)

DEVICE = "cuda"

BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "4096"))
THRESHOLD = int(os.environ.get("THRESHOLD", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "512"))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", "0.05"))

START_LAYER = int(os.environ.get("START_LAYER", "14"))
END_LAYER = int(os.environ.get("END_LAYER", "16"))

MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))

# 你要求的提速配置
PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1")
)
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8")
)

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))

MAX_TRAIN_LINES = int(os.environ.get("MAX_TRAIN_LINES", "40000"))
MAX_VALID_LINES = int(os.environ.get("MAX_VALID_LINES", "5000"))

SEED = 42
NUM_PROC = 12


# ============================================================
# 兼容不同 transformers 版本
# ============================================================

def build_training_arguments():
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "output_dir": SAVE_DIR,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "bf16": True,

        # 你要求：关闭 gradient checkpointing
        "gradient_checkpointing": os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1",

        # 你要求：logging/eval/save 频率调整
        "logging_steps": 50,
        "save_steps": 1000,
        "save_total_limit": 1,

        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "report_to": "none",
        "dataloader_num_workers": 8,
        "remove_unused_columns": False,
    }

    if "warmup_steps" in supported:
        kwargs["warmup_steps"] = 30
    elif "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = 0.03

    if "lr_scheduler_type" in supported:
        kwargs["lr_scheduler_type"] = "cosine"

    if "logging_strategy" in supported:
        kwargs["logging_strategy"] = "steps"

    if "save_strategy" in supported:
        kwargs["save_strategy"] = "steps"

    if "save_only_model" in supported:
        kwargs["save_only_model"] = True

    # 训练阶段关闭 eval，避免 4096 长度验证时 logits OOM
    # 训练完成后用 compare_laoppe_score_bias.py 单独评估 PPL

    final_kwargs = {}
    skipped = {}

    for k, v in kwargs.items():
        if k in supported:
            final_kwargs[k] = v
        else:
            skipped[k] = v

    print("=" * 80)
    print("TrainingArguments 参数兼容性检查")
    print("=" * 80)
    print("将使用参数:")
    for k in sorted(final_kwargs.keys()):
        print(f"  {k}: {final_kwargs[k]}")

    if skipped:
        print("当前 transformers 不支持，已跳过:")
        for k in sorted(skipped.keys()):
            print(f"  {k}: {skipped[k]}")
    else:
        print("没有跳过参数")
    print("=" * 80)

    return TrainingArguments(**final_kwargs)


class LAOPPEMetricCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or model is None:
            return

        metrics = collect_laoppe_metrics(model)
        if metrics:
            logs.update(metrics)


# ============================================================
# 数据处理
# ============================================================

def tokenize_fn(examples, tokenizer):
    all_input_ids = []

    for text in examples["text"]:
        text = text.strip()
        if len(text) == 0:
            continue

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) == 0:
            continue

        ids.append(tokenizer.eos_token_id)
        all_input_ids.append(ids)

    return {"input_ids": all_input_ids}


def group_texts(examples, block_size):
    concatenated_ids = []

    for ids in examples["input_ids"]:
        concatenated_ids.extend(ids)

    total_length = len(concatenated_ids)
    total_length = (total_length // block_size) * block_size

    if total_length == 0:
        return {
            "input_ids": [],
            "labels": [],
        }

    input_ids = [
        concatenated_ids[i: i + block_size]
        for i in range(0, total_length, block_size)
    ]

    labels = [x.copy() for x in input_ids]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_lm_dataset(tokenizer):
    print("🚀 加载 WikiText 数据...")

    raw_ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_DATA_PATH,
            "validation": VALID_DATA_PATH,
        }
    )

    train_raw = raw_ds["train"].filter(
        lambda x: len(x["text"].strip()) > 0
    )
    valid_raw = raw_ds["validation"].filter(
        lambda x: len(x["text"].strip()) > 0
    )

    train_raw = train_raw.shuffle(seed=SEED)

    if MAX_TRAIN_LINES > 0:
        train_raw = train_raw.select(
            range(min(MAX_TRAIN_LINES, len(train_raw)))
        )

    if MAX_VALID_LINES > 0:
        valid_raw = valid_raw.select(
            range(min(MAX_VALID_LINES, len(valid_raw)))
        )

    print(f"train 原始行数: {len(train_raw)}")
    print(f"valid 原始行数: {len(valid_raw)}")

    print("⚡ tokenize train...")
    tokenized_train = train_raw.map(
        tokenize_fn,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text"],
        desc="Tokenizing train",
    )

    print("⚡ tokenize valid...")
    tokenized_valid = valid_raw.map(
        tokenize_fn,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text"],
        desc="Tokenizing valid",
    )

    print(f"⚡ packing train into block_size={BLOCK_SIZE}...")
    train_ds = tokenized_train.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"block_size": BLOCK_SIZE},
        desc="Grouping train",
    )

    print(f"⚡ packing valid into block_size={BLOCK_SIZE}...")
    valid_ds = tokenized_valid.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"block_size": BLOCK_SIZE},
        desc="Grouping valid",
    )

    print("=" * 80)
    print(f"train blocks: {len(train_ds)}")
    print(f"valid blocks: {len(valid_ds)}")
    print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    print(f"估计 train tokens: {len(train_ds) * BLOCK_SIZE}")
    print(f"估计 valid tokens: {len(valid_ds) * BLOCK_SIZE}")
    print("=" * 80)

    return train_ds, valid_ds


# ============================================================
# 模型处理
# ============================================================

def load_base_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    return model


def freeze_only_laoppe_params(model):
    for name, param in model.named_parameters():
        if "laoppe_bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_parameters(model):
    total = 0
    trainable = 0

    print("=" * 80)
    print("可训练参数列表")
    print("=" * 80)

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        if p.requires_grad:
            trainable += n
            print(f"TRAINABLE: {name} | {n:,}")

    ratio = trainable / total * 100

    print("=" * 80)
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    print(f"可训练比例: {ratio:.8f}%")
    print("=" * 80)


def load_model():
    print("🚀 加载基础 LLaMA 模型...")
    model = load_base_model()

    print("🚀 接入 Distance-Gated Residual LA-OPPE Score Bias")
    model = replace_llama_attention_score_bias(
        model,
        start_layer=START_LAYER,
        end_layer=END_LAYER,
        threshold=THRESHOLD,
        temperature=TEMPERATURE,
        max_len=131072,
        alpha_max=ALPHA_MAX,
    )

    if LOAD_STATE_DICT:
        print(f"🔁 加载已有 full_state_dict: {LOAD_STATE_DICT}")
        sd = torch.load(LOAD_STATE_DICT, map_location="cpu")
        incompatible = model.load_state_dict(sd, strict=False)

        print(f"missing keys: {len(incompatible.missing_keys)}")
        print(f"unexpected keys: {len(incompatible.unexpected_keys)}")

        if incompatible.missing_keys:
            print("前 20 个 missing:")
            for k in incompatible.missing_keys[:20]:
                print("  MISSING:", k)

        if incompatible.unexpected_keys:
            print("前 20 个 unexpected:")
            for k in incompatible.unexpected_keys[:20]:
                print("  UNEXPECTED:", k)

        del sd

    model.config.use_cache = False

    freeze_only_laoppe_params(model)

    # 你要求：关掉 checkpointing，这里不启用
    if os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1":
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.to(DEVICE)

    print_trainable_parameters(model)

    return model


def save_model_safely(trainer, tokenizer):
    print("💾 保存模型...")

    try:
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    except Exception:
        unwrapped_model = trainer.model

    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        unwrapped_model.save_pretrained(
            SAVE_DIR,
            safe_serialization=True,
        )
    except TypeError:
        unwrapped_model.save_pretrained(SAVE_DIR)

    tokenizer.save_pretrained(SAVE_DIR)

    full_state_path = os.path.join(SAVE_DIR, "full_state_dict.pt")
    state_dict = unwrapped_model.state_dict()

    print("=" * 80)
    print("保存前 state_dict 检查")
    print("=" * 80)
    print("是否包含 lm_head.weight:", "lm_head.weight" in state_dict)

    laoppe_keys = [
        k for k in state_dict.keys()
        if "laoppe_bias" in k
    ]

    print("laoppe_bias 参数数量:", len(laoppe_keys))
    for k in laoppe_keys[:20]:
        print(" ", k)

    torch.save(state_dict, full_state_path)

    print(f"✅ HuggingFace 模型保存在: {SAVE_DIR}")
    print(f"✅ 完整 state_dict 保存在: {full_state_path}")

    del state_dict
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_trainer(model, train_args, train_ds, valid_ds, tokenizer):
    sig = inspect.signature(Trainer.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        # 训练阶段不传 eval_dataset，避免 Trainer 自动评估 OOM
        # "eval_dataset": valid_ds,
        "data_collator": default_data_collator,
        "callbacks": [LAOPPEMetricCallback()],
    }

    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


# ============================================================
# 主函数
# ============================================================

def main():
    multiprocessing.freeze_support()
    set_seed(SEED)

    print("=" * 80)
    print("Distance-Gated Residual LA-OPPE Score Bias 训练")
    print("=" * 80)
    print(f"BASE_MODEL_PATH = {BASE_MODEL_PATH}")
    print(f"LOAD_STATE_DICT = {LOAD_STATE_DICT}")
    print(f"SAVE_DIR        = {SAVE_DIR}")
    print(f"BLOCK_SIZE      = {BLOCK_SIZE}")
    print(f"THRESHOLD       = {THRESHOLD}")
    print(f"START_LAYER     = {START_LAYER}")
    print(f"END_LAYER       = {END_LAYER}")
    print(f"MAX_STEPS       = {MAX_STEPS}")
    print(f"LR              = {LEARNING_RATE}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    train_ds, valid_ds = build_lm_dataset(tokenizer)

    model = load_model()

    train_args = build_training_arguments()

    trainer = build_trainer(
        model=model,
        train_args=train_args,
        train_ds=train_ds,
        valid_ds=valid_ds,
        tokenizer=tokenizer,
    )

    print("✅ 开始训练...")
    trainer.train()

    save_model_safely(trainer, tokenizer)

    print("✅ 训练完成")


if __name__ == "__main__":
    main()
