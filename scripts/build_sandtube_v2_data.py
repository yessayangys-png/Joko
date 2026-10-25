from __future__ import annotations

import json
import random
from pathlib import Path

from behavior_examples import BEHAVIOR
from kbbi_lexicon import KBBI_ROWS

SEED = 2026
TRAIN = Path("data/processed/sandtube_ai_v2_train.jsonl")
VALID = Path("data/processed/sandtube_ai_v2_valid.jsonl")


def math_rows() -> list[dict[str, str]]:
    rows = []
    operations = [
        ("ditambah", lambda a, b: a + b),
        ("dikurangi", lambda a, b: a - b),
        ("dikali", lambda a, b: a * b),
    ]
    for a in range(2, 22):
        for b in range(2, 12):
            for word, fn in operations:
                result = fn(a, b)
                rows.append({"prompt": f"Berapakah {a} {word} {b}? Jawab singkat.", "response": f"{a} {word} {b} adalah {result}.", "category": "math", "source": "deterministic_math"})
                if len(rows) >= 300:
                    return rows
    return rows


def perspective_rows() -> list[dict[str, str]]:
    pairs = [
        ("Siapa nama ibu saya? Saya belum pernah memberi tahu kamu.", "Saya tidak tahu karena Anda belum pernah memberi tahu saya. Saya tidak akan menebak informasi pribadi itu."),
        ("Apa nama kamu?", "Saya SandTube-Ai, asisten berbahasa Indonesia yang dapat membantu menjawab pertanyaan."),
        ("Apa yang kamu ketahui tentang saya?", "Saya hanya mengetahui informasi yang Anda berikan dalam percakapan ini. Saya tidak akan menebak data pribadi Anda."),
        ("Ringkas dokumen yang belum saya kirim.", "Saya belum menerima dokumennya, jadi belum bisa merangkumnya. Silakan kirim atau tempelkan teksnya."),
        ("Apa isi gambar yang belum saya unggah?", "Saya belum menerima gambarnya. Silakan unggah gambar tersebut agar saya dapat membantu mendeskripsikannya."),
        ("Berikan DOI untuk penelitian yang belum saya sebutkan.", "Saya tidak akan membuat DOI palsu. Kirim judul, klaim, atau sumber penelitian yang ingin diperiksa."),
        ("Apa itu Java?", "Java dapat merujuk pada bahasa pemrograman atau Pulau Jawa. Jika maksud Anda bahasa pemrograman, beri tahu konteksnya."),
        ("Apa yang harus dilakukan jika pertanyaan saya kurang jelas?", "Saya sebaiknya meminta klarifikasi atau menjelaskan asumsi yang diperlukan, bukan mengarang konteks."),
    ]
    rows = []
    for prompt, response in pairs:
        rows.append({"prompt": prompt, "response": response, "category": "behavior", "source": "curated_v2"})
        rows.append({"prompt": prompt.replace("?", "? Tolong jawab dengan jujur."), "response": response, "category": "behavior", "source": "curated_v2_paraphrase"})
    return rows


def main() -> None:
    rng = random.Random(SEED)
    rows = math_rows() + perspective_rows()
    rows += [{"prompt": str(prompt).strip(), "response": str(response).strip(), "category": "curated", "source": "repo_behavior"} for prompt, response in BEHAVIOR]
    rows += [{"prompt": str(row["prompt"]).strip(), "response": str(row["response"]).strip(), "category": "knowledge", "source": "repo_kbbi"} for row in KBBI_ROWS]
    seen = set()
    unique = []
    for row in rows:
        key = (row["prompt"].casefold(), row["response"].casefold())
        if key not in seen and len(row["prompt"]) >= 5 and len(row["response"]) >= 8:
            seen.add(key)
            unique.append(row)
    rng.shuffle(unique)
    valid_count = max(40, len(unique) // 10)
    valid = unique[:valid_count]
    train = unique[valid_count:]
    for path, items in ((TRAIN, train), (VALID, valid)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")
    report = {"total": len(unique), "train": len(train), "valid": len(valid), "categories": {}}
    for row in unique:
        report["categories"][row["category"]] = report["categories"].get(row["category"], 0) + 1
    Path("data/processed/sandtube_ai_v2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
