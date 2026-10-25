from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "SHA256SUMS.json").read_text(encoding="utf-8"))
OUTPUT = HERE / MANIFEST["source_weight"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {OUTPUT}")
    expected_size = MANIFEST["source_weight_size_bytes"]
    written = 0
    with OUTPUT.open("xb") as out:
        for part in MANIFEST["split_into"]:
            path = HERE / part["path"]
            if path.stat().st_size != part["size_bytes"]:
                raise SystemExit(f"Unexpected shard size: {path}")
            if sha256(path) != part["sha256"]:
                raise SystemExit(f"Shard checksum mismatch: {path}")
            with path.open("rb") as src:
                for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
                    out.write(chunk)
                    written += len(chunk)
    if written != expected_size or sha256(OUTPUT) != MANIFEST["source_weight_sha256"]:
        OUTPUT.unlink(missing_ok=True)
        raise SystemExit("Reconstructed model failed size/SHA-256 verification; output removed.")
    print(f"Restored and verified {OUTPUT} ({written} bytes)")


if __name__ == "__main__":
    main()
