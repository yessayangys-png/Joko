from __future__ import annotations

import os
from typing import Dict

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache/transformers")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

MODEL_NAME = os.getenv("MODEL_NAME", "izrai4/SandTube_Chat")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/indonesian_training.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/sandtube_indonesia")
MAX_STEPS = int(os.getenv("MAX_STEPS", "300"))
MAX_DATA_ROWS = int(os.getenv("MAX_DATA_ROWS", "5000"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))


def build_chat_text(example: Dict[str, str]) -> str:
    prompt = str(example.get("prompt", "")).strip()
    response = str(example.get("response", "")).strip()
    messages = [
        {"role": "system", "content": "Kamu adalah AI berbahasa Indonesia yang menggunakan kosakata baku sesuai KBBI, jelas, ringkas, dan natural."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return messages


def tokenize_fn(example, tokenizer):
    messages = build_chat_text(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=768,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    print(f"Mempersiapkan model: {MODEL_NAME}")
    print(f"Membaca data: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}. Jalankan python download_indonesia_datasets.py terlebih dahulu.")

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(len(dataset), MAX_DATA_ROWS)))

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
        model.config.use_cache = False
        fp16_flag = False
        optim_name = "adamw_torch"

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=False,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        fp16=fp16_flag,
        bf16=False,
        max_steps=MAX_STEPS,
        logging_steps=20,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=20,
        report_to=[],
        optim=optim_name,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Memulai fine-tuning ringan (LoRA) untuk chat Indonesia...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model selesai disimpan di: {OUTPUT_DIR}")
    print("Ukuran target akhir tetap dalam rentang 4-8 GB karena model menggunakan LoRA + kuantisasi 4-bit.")


if __name__ == "__main__":
    main()
