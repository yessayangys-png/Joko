from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chat_sandtube import load_chat_model, get_reply


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def score_case(case: dict, answer: str) -> tuple[bool, list[str]]:
    text = norm(answer)
    reasons: list[str] = []
    required = [norm(x) for x in case.get("must_contain_any", [])]
    forbidden = [norm(x) for x in case.get("must_not_contain", [])]
    if required and not any(x in text for x in required):
        reasons.append("missing_expected_signal")
    found_forbidden = [x for x in forbidden if x in text]
    if found_forbidden:
        reasons.append("contains_forbidden:" + ",".join(found_forbidden))
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi SandTube-Ai dengan benchmark JSON.")
    parser.add_argument("--benchmark", default="evaluation_sandtube_ai_v2.json")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    if args.limit > 0:
        cases = cases[: args.limit]
    tokenizer, model = load_chat_model()
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            answer = get_reply(tokenizer, model, [], case["prompt"])
            passed, reasons = score_case(case, answer)
            error = None
        except Exception as exc:  # keep the remaining benchmark runnable
            answer, passed, reasons = "", False, ["exception"]
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "answer": answer,
                "passed": passed,
                "reasons": reasons,
                "seconds": round(time.perf_counter() - started, 3),
                "error": error,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    passed = sum(item["passed"] for item in results)
    by_category: dict[str, dict[str, int]] = {}
    for item in results:
        bucket = by_category.setdefault(item["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(item["passed"])
    report = {
        "benchmark": args.benchmark,
        "cases": len(results),
        "passed": passed,
        "accuracy": round(passed / len(results), 4) if results else 0.0,
        "by_category": by_category,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("benchmark", "cases", "passed", "accuracy", "by_category")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
