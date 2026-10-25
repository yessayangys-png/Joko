from __future__ import annotations

import json
import random
import re
from pathlib import Path

from langdetect import DetectorFactory, detect_langs
from behavior_examples import BEHAVIOR
from kbbi_lexicon import KBBI_ROWS

DetectorFactory.seed = 0
RAW = Path("data/huggingface/indonesian_raw_normalized.jsonl")
CLEAN = Path("data/huggingface/indonesian_clean.jsonl")
TRAIN = Path("data/processed/sandtube_ai_train.jsonl")
VALID = Path("data/processed/sandtube_ai_valid.jsonl")
SEED = 2026
ROLE_PAIR = re.compile(r"(?:Pengguna|User|Human)\s*:\s*(.*?)\s*(?:Asisten|Assistant|GPT)\s*:\s*(.*)", re.I | re.S)
ROLE_START = re.compile(r"^\s*(?:Pengguna|User|Human|Asisten|Assistant|GPT)\s*:\s*", re.I)
SAFE_PROMPT = re.compile(
    r"(?:cara terbaik untuk belajar|bagaimana saya bisa belajar|strategi belajar|jadwal belajar|"
    r"tips? .*belajar|tips? .*menulis|blokir penulis|surat lamaran|menulis resume|"
    r"jalur karier apa|strategi terbaik untuk berolahraga|patung .*salju|"
    r"aktivitas .*luar ruangan|aktivitas .*bersenang-senang|apa yang harus saya lakukan malam ini untuk bersenang-senang|"
    r"tips? mencari pekerjaan|tip terbaik untuk mencari pekerjaan)", re.I
)
RISK = re.compile(
    r"\b(berapa|kapan|siapa|tahun|tanggal|harga|biaya|gaji|pemilu|obat|gejala|penyakit|hukum|pasal|"
    r"investasi|saham|kripto|dokter|diagnosis|ras|agama|usia|umur|aman sekarang|peraturan|variasi|"
    r"kloroform|kartu kredit|paspor|visa|terbaru|terbesar|tertua|pertama|covid|depresi klinis|"
    r"merasa agak sedih|membuat saya merasa)\b", re.I
)


def clean_text(value: str) -> str:
    value = value.replace("\r", "\n").strip()
    value = ROLE_START.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def language_id(text: str) -> tuple[str, float]:
    try:
        result = detect_langs(text[:1200])
        return result[0].lang, float(result[0].prob)
    except Exception:
        return "unknown", 0.0


def extract_pair(row: dict) -> tuple[str, str] | None:
    prompt, response = str(row.get("prompt", "")), str(row.get("response", ""))
    # Some HF rows store an Indonesian translation as a user/assistant pair in response.
    match = ROLE_PAIR.search(response)
    if match:
        return clean_text(match.group(1)), clean_text(match.group(2))
    # Drop malformed prompt fields that already contain a full source conversation.
    if re.search(r"(?:^|\n)\s*(?:User|Pengguna)\s*:", prompt, re.I) and re.search(r"(?:^|\n)\s*(?:Assistant|Asisten)\s*:", prompt, re.I):
        return None
    if ROLE_PAIR.search(prompt):
        return None
    return clean_text(prompt), clean_text(response)


def main():
    rng = random.Random(SEED)
    rejects: dict[str, int] = {}
    candidates, seen = [], set()

    def reject(reason: str):
        rejects[reason] = rejects.get(reason, 0) + 1

    for line in RAW.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            reject("invalid_json"); continue
        source = row.get("source", "")
        # The downloaded INTISARI sample and broader Cahya trivia contain
        # low-quality/fabricated answers; retain only the explicit safe whitelist below.
        if source == "INTISARI/intisari-indonesian-chat-v5":
            reject("excluded_intisari_quality"); continue
        pair = extract_pair(row)
        if not pair:
            reject("malformed_embedded_conversation"); continue
        prompt, response = pair
        if not prompt or not response or len(prompt) < 5 or len(response) < 12 or len(prompt) > 1200 or len(response) > 2400:
            reject("empty_or_length"); continue
        if SAFE_PROMPT.search(prompt) is None or RISK.search(prompt) or RISK.search(response):
            reject("outside_low_risk_whitelist"); continue
        if re.search(r"\bperbaiki\b.*\bberikut\b", prompt, re.I) or re.search(r"\bPT\s+[A-Z][\w.-]+", response):
            reject("unsupported_specific_details"); continue
        if re.search(r"\b(?:Pengguna|Asisten|User|Assistant)\s*:", response, re.I) or re.search(r"\bvariasi\s+\d+", prompt, re.I):
            reject("role_leak_or_synthetic_variant"); continue
        if re.search(r"\d", response):
            reject("numeric_claims"); continue
        lp, pp = language_id(prompt); lr, pr = language_id(response)
        if lp != "id" or pp < 0.62 or lr != "id" or pr < 0.62:
            reject("language_filter"); continue
        key = (re.sub(r"\W+", " ", prompt.casefold()).strip(), re.sub(r"\W+", " ", response.casefold()).strip())
        if key in seen:
            reject("duplicate"); continue
        seen.add(key)
        candidates.append({"prompt": prompt, "response": response, "source": source})

    rng.shuffle(candidates)
    hf_rows = candidates[:80]
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN.open("w", encoding="utf-8") as f:
        for row in hf_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    behavior = [{"prompt": clean_text(q), "response": clean_text(a), "source": "curated_behavior"} for q, a in BEHAVIOR]
    held_out_behavior = {0, 4, 8, 12, 20, 23, 30, 34}
    behavior_train = [r for i, r in enumerate(behavior) if i not in held_out_behavior]
    behavior_valid = [r for i, r in enumerate(behavior) if i in held_out_behavior]
    kbbi = [{"prompt": clean_text(r["prompt"]), "response": clean_text(r["response"]), "source": "repo_kbbi_lexicon"} for r in KBBI_ROWS]
    rng.shuffle(kbbi)
    kbbi_valid = kbbi[:max(2, len(kbbi) // 8)]
    kbbi_train = kbbi[len(kbbi_valid):]
    rng.shuffle(hf_rows)
    hf_valid = hf_rows[:max(2, len(hf_rows) // 6)] if hf_rows else []
    hf_train = hf_rows[len(hf_valid):]

    train_rows = hf_train + kbbi_train * 2 + behavior_train * 4
    valid_rows = hf_valid + kbbi_valid + behavior_valid
    rng.shuffle(train_rows); rng.shuffle(valid_rows)
    for path, rows in ((TRAIN, train_rows), (VALID, valid_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False) + "\n")
    report = {
        "raw_huggingface_rows_downloaded": sum(1 for _ in RAW.open(encoding="utf-8")),
        "hf_rows_passing_low_risk_language_filters": len(hf_rows),
        "hf_rows_used_train": len(hf_train),
        "hf_rows_held_out": len(hf_valid),
        "hf_filter_rejections": rejects,
        "repo_kbbi_train_unique": len(kbbi_train),
        "repo_kbbi_train_repeated_2x": 2 * len(kbbi_train),
        "curated_behavior_train_unique": len(behavior_train),
        "curated_behavior_train_repeated_4x": 4 * len(behavior_train),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "validation_sources": {"hf": len(hf_valid), "kbbi": len(kbbi_valid), "held_out_behavior": len(behavior_valid)},
    }
    Path("data/huggingface/filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {TRAIN} and {VALID}")


if __name__ == "__main__":
    main()
