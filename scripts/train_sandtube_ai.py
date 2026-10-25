from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache/transformers")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = os.getenv("MODEL_NAME", "izrai4/SandTube_Chat")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/sandtube_ai_train.jsonl")
VALID_PATH = os.getenv("VALID_PATH", "data/processed/sandtube_ai_valid.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/SandTube-Ai-LoRA")
MAX_STEPS = int(os.getenv("MAX_STEPS", "100"))
MAX_DATA_ROWS = int(os.getenv("MAX_DATA_ROWS", "0"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "384"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "7e-5"))
MAX_ADAPTER_BYTES = int(os.getenv("MAX_ADAPTER_BYTES", "9034304"))
SYSTEM_PROMPT = (
    "Kamu adalah SandTube-Ai, asisten berbahasa Indonesia yang jelas dan ringkas. "
    "Jawab langsung berdasarkan informasi yang tersedia. Jangan mengarang fakta. "
    "Jika tidak tahu atau konteks kurang, katakan dengan jujur dan minta informasi yang diperlukan."
)


def get_input_ids(encoded: Any) -> list[int]:
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def tokenize_example(example: dict[str, Any], tokenizer: Any) -> dict[str, list[int]]:
    prompt = str(example.get("prompt", "")).strip()
    response = str(example.get("response", "")).strip()
    user_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    full_messages = user_messages + [{"role": "assistant", "content": response}]
    prefix_ids = get_input_ids(
        tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = get_input_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat template tidak menghasilkan prefix assistant yang sama; label masking dibatalkan.")
    answer_ids = full_ids[len(prefix_ids) :]
    if not answer_ids:
        raise ValueError("Contoh tidak memiliki token jawaban assistant.")

    # Preserve the entire answer when possible; trim only the oldest prompt tokens.
    if len(answer_ids) >= MAX_LENGTH - 8:
        answer_ids = answer_ids[: MAX_LENGTH - 8]
    max_prefix = MAX_LENGTH - len(answer_ids)
    if len(prefix_ids) > max_prefix:
        prefix_ids = prefix_ids[-max_prefix:]
    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def main() -> None:
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset training tidak ditemukan: {DATA_PATH}; jalankan prepare_honesty_data.py terlebih dahulu.")
    if not Path(VALID_PATH).exists():
        raise FileNotFoundError(f"Dataset validasi tidak ditemukan: {VALID_PATH}; jalankan prepare_honesty_data.py terlebih dahulu.")

    print(f"Memuat data train: {DATA_PATH}")
    train_ds = load_dataset("json", data_files=DATA_PATH, split="train")
    valid_ds = load_dataset("json", data_files=VALID_PATH, split="train")
    if MAX_DATA_ROWS > 0:
        train_ds = train_ds.shuffle(seed=42).select(range(min(len(train_ds), MAX_DATA_ROWS)))
    print(f"Baris train={len(train_ds)}, validasi={len(valid_ds)}, max_length={MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        fp16_flag = torch.cuda.get_device_capability()[0] >= 8
        optim_name = "paged_adamw_8bit"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cpu"},
            trust_remote_code=True,
            dtype=torch.float32,
        )
        fp16_flag = False
        optim_name = "adamw_torch"

    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def map_one(example: dict[str, Any]) -> dict[str, list[int]]:
        return tokenize_example(example, tokenizer)

    train_tok = train_ds.map(map_one, remove_columns=train_ds.column_names)
    valid_tok = valid_ds.map(map_one, remove_columns=valid_ds.column_names)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=fp16_flag,
        bf16=False,
        max_steps=MAX_STEPS,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=min(4, max(1, MAX_STEPS // 10)),
        report_to="none",
        optim=optim_name,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
    )
    print("Memulai fine-tuning LoRA dengan loss hanya pada jawaban assistant.", flush=True)
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    adapter_file = Path(OUTPUT_DIR) / "adapter_model.safetensors"
    adapter_size = adapter_file.stat().st_size
    if adapter_size > MAX_ADAPTER_BYTES:
        adapter_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"Adapter baru ({adapter_size} bytes) melampaui batas ukuran lama "
            f"({MAX_ADAPTER_BYTES} bytes); bobot adapter yang terlalu besar dihapus."
        )
    metrics = {"train": train_result.metrics, "eval": eval_result, "method": "LoRA on CPU" if not use_cuda else "QLoRA 4-bit", "adapter_bytes": adapter_size, "adapter_size_limit_bytes": MAX_ADAPTER_BYTES, "adapter_rank": 8}
    with (Path(OUTPUT_DIR) / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float))
    print(f"Adapter baru tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
