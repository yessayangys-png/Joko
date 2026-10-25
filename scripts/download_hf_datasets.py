from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache/transformers")

from datasets import load_dataset

DATASETS = [
    "cahya/instructions_indonesian",
    "LorthGyu/indonesian-instructions",
    "INTISARI/intisari-indonesian-chat-v5",
]
OUT = Path("data/huggingface/indonesian_raw_normalized.jsonl")


def text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(text(x) for x in value)
    if isinstance(value, dict):
        return str(value.get("content", value.get("value", ""))).strip()
    return str(value).strip()


def normalize(row):
    pairs = [
        ("prompt", "response"),
        ("instruction", "output"),
        ("instruction", "response"),
        ("question", "answer"),
        ("query", "response"),
        ("input", "output"),
    ]
    for p, r in pairs:
        if p in row and r in row:
            prompt, response = text(row[p]), text(row[r])
            if prompt and response:
                return prompt, response
    if "conversations" in row:
        conv = row["conversations"]
        if isinstance(conv, list):
            user = next((text(m.get("value", m.get("content", ""))) for m in conv if str(m.get("from", m.get("role", ""))).lower() in {"human", "user"}), "")
            assistant = next((text(m.get("value", m.get("content", ""))) for m in reversed(conv) if str(m.get("from", m.get("role", ""))).lower() in {"gpt", "assistant"}), "")
            if user and assistant:
                return user, assistant
    if "text" in row and "label" in row:
        label = row["label"]
        if isinstance(label, list):
            label = label[0] if label else ""
        prompt, response = text(row["text"]), text(label)
        if prompt and response:
            return prompt, response
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    stats = {}
    for name in DATASETS:
        print(f"Mengunduh ulang {name} dari Hugging Face...", flush=True)
        try:
            ds = load_dataset(
                name,
                split="train[:2000]",
                download_mode="force_redownload",
            )
            count = 0
            for item in ds:
                pair = normalize(item)
                if pair:
                    prompt, response = pair
                    records.append({"prompt": prompt, "response": response, "source": name})
                    count += 1
            stats[name] = {"rows_seen": len(ds), "pairs_extracted": count}
            print(f"  didapat {count} pasangan dari {len(ds)} baris sampel", flush=True)
        except Exception as exc:
            stats[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  dilewati: {type(exc).__name__}: {exc}", flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = OUT.with_name("download_manifest.json")
    manifest.write_text(json.dumps({"datasets": stats, "normalized_rows": len(records)}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError("Tidak ada data yang berhasil diunduh; tidak memulai training.")
    print(f"Tersimpan: {OUT} ({len(records)} baris); manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
