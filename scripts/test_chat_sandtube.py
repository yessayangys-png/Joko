from __future__ import annotations

import chat_sandtube


def test_get_reply_includes_current_question(monkeypatch):
    captured = []

    def fake_generate(tokenizer, model, messages):
        captured.append(messages)
        return "Jawaban valid"

    monkeypatch.setattr(chat_sandtube, "generate", fake_generate)
    answer = chat_sandtube.get_reply(
        tokenizer=object(),
        model=object(),
        history=[{"role": "user", "content": "Pertanyaan lama"}, {"role": "assistant", "content": "Jawaban lama"}],
        question="Pertanyaan terbaru",
    )
    assert answer == "Jawaban valid"
    assert captured[0][-1] == {"role": "user", "content": "Pertanyaan terbaru"}


if __name__ == "__main__":
    class MonkeyPatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_get_reply_includes_current_question(MonkeyPatch())
    print("PASS: current question is included")
