from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache/transformers")

from datasets import load_dataset
from kbbi_lexicon import KBBI_ROWS

DATASET_CANDIDATES = [
    "cahya/instructions_indonesian",
    "LorthGyu/indonesian-instructions",
    "INTISARI/intisari-indonesian-chat-v5",
]

OUTPUT_PATH = Path("data/processed/indonesian_training.jsonl")


def ensure_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value).strip()


def normalize_record(record):
    if "prompt" in record and "response" in record:
        return {"prompt": ensure_text(record["prompt"]), "response": ensure_text(record["response"])}
    if "instruction" in record and "output" in record:
        return {"prompt": ensure_text(record["instruction"]), "response": ensure_text(record["output"])}
    if "text" in record and "label" in record:
        label = record["label"]
        response = label[0] if isinstance(label, list) else label
        return {"prompt": ensure_text(record["text"]), "response": ensure_text(response)}
    if "conversations" in record:
        conversations = record["conversations"]
        if isinstance(conversations, list) and len(conversations) >= 2:
            prompt = ensure_text(conversations[0].get("value", ""))
            response = ensure_text(conversations[-1].get("value", ""))
            if prompt and response:
                return {"prompt": prompt, "response": response}
    return None


def build_fallback_rows():
    return [
        {"prompt": "Halo, apa kabar?", "response": "Halo! Saya baik, terima kasih. Saya siap membantu dengan pertanyaan Anda dalam bahasa Indonesia."},
        {"prompt": "Siapa kamu?", "response": "Saya adalah model AI yang dibuat untuk membantu menjawab pertanyaan dan berbicara dalam bahasa Indonesia."},
        {"prompt": "Apa tujuanmu dibuat?", "response": "Tujuan saya dibuat adalah untuk membantu manusia menjawab pertanyaan, membuat teks, dan berdiskusi dalam bahasa Indonesia."},
        {"prompt": "Bahasa Indonesia yang kamu pakai itu yang benar?", "response": "Saya menggunakan Bahasa Indonesia yang baku dan sederhana agar mudah dipahami."},
        {"prompt": "Jelaskan apa itu AI.", "response": "AI atau kecerdasan buatan adalah bidang komputer yang memungkinkan mesin belajar dari data dan membuat keputusan atau jawaban."},
        {"prompt": "Bagaimana cara belajar coding?", "response": "Mulai dari bahasa pemrograman dasar, pahami logika, lalu latihan setiap hari dengan mini project."},
        {"prompt": "Apa itu machine learning?", "response": "Machine learning adalah teknik di mana komputer belajar pola dari data tanpa harus diprogram secara eksplisit untuk setiap kasus."},
        {"prompt": "Kenapa model AI harus sering dilatih?", "response": "Agar model dapat menyesuaikan diri dengan data baru, pola baru, dan gaya bahasa yang lebih akurat."},
        {"prompt": "Bisa bantu saya menulis email formal?", "response": "Tentu, saya bisa membantu menulis email formal dengan bahasa yang sopan dan jelas."},
        {"prompt": "Apa yang dimaksud dengan fine-tuning?", "response": "Fine-tuning adalah proses melatih ulang model yang sudah ada dengan data tertentu agar lebih sesuai untuk tugas tertentu."},
    ]


def is_indonesian_text(text: str) -> bool:
    lower = text.lower()
    markers = [
        "bahasa indonesia",
        "indonesia",
        "indonesian",
        "kamu",
        "saya",
        "apakah",
        "bagaimana",
        "tolong",
        "halo",
        "kami",
        "apa",
        "yang",
        "ini",
        "itu",
        "di sini",
    ]
    return any(marker in lower for marker in markers)


def build_kbbi_rows():
    rows = []
    for item in KBBI_ROWS:
        prompt = item["prompt"].strip()
        response = item["response"].strip()
        if prompt and response:
            rows.append({"prompt": prompt, "response": response})
    return rows


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = build_kbbi_rows()

    for dataset_name in DATASET_CANDIDATES:
        print(f"Mencoba dataset: {dataset_name}")
        try:
            ds = load_dataset(dataset_name, split="train[:800]")
        except Exception as exc:
            print(f"  -> dilewati: {type(exc).__name__}: {exc}")
            continue

        for item in ds:
            normalized = normalize_record(item)
            if not normalized:
                continue
            prompt = normalized["prompt"].strip()
            response = normalized["response"].strip()
            if not prompt or not response:
                continue
            if not is_indonesian_text(prompt + " " + response):
                continue
            rows.append({"prompt": prompt, "response": response})

    if not rows:
        rows = build_fallback_rows()
        print("Data dari Hugging Face tidak tersedia. Menggunakan fallback dataset Indonesia untuk melanjutkan.")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Data berhasil disimpan ke {OUTPUT_PATH}")
    print(f"Jumlah baris: {len(rows)}")


if __name__ == "__main__":
    main()
