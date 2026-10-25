from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "izrai4/SandTube_Chat")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "output/SandTube-Ai-LoRA")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "250"))
SYSTEM_PROMPT = (
    "Kamu adalah SandTube-Ai, asisten berbahasa Indonesia yang jelas dan ringkas. Jawab langsung pertanyaan pengguna. "
    "Jangan menyalin atau mengulang pertanyaan pengguna sebagai jawaban. Jangan mengarang fakta. "
    "Jika tidak tahu atau konteks kurang, katakan dengan jujur dan minta informasi yang diperlukan."
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold().strip(" .?!")


def looks_like_echo(answer: str, question: str) -> bool:
    a, q = normalize(answer), normalize(question)
    if not a:
        return True
    if a == q:
        return True
    q_words, a_words = q.split(), a.split()
    if len(a_words) >= 3 and a in q:
        return True
    if len(a_words) >= 4 and len(set(a_words) - set(q_words)) == 0:
        return True
    if len(q_words) >= 6 and len(a_words) >= 6:
        overlap = len(set(q_words) & set(a_words)) / max(1, len(set(q_words) | set(a_words)))
        return overlap > 0.88
    return False


def load_chat_model():
    print(f"Memuat tokenizer dan model dasar {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        trust_remote_code=True,
    )
    if Path(ADAPTER_PATH).exists():
        print(f"Memasang adapter LoRA: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print(f"Adapter tidak ditemukan di {ADAPTER_PATH}; menggunakan model dasar saja.")
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()


def get_reply(tokenizer, model, history, question):
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": question}]
    )
    answer = generate(tokenizer, model, messages)
    if looks_like_echo(answer, question):
        stricter_system = SYSTEM_PROMPT + " Instruksi penting: keluaran wajib berupa jawaban, bukan pengulangan atau parafrasa pertanyaan."
        answer = generate(
            tokenizer,
            model,
            [{"role": "system", "content": stricter_system}]
            + history
            + [{"role": "user", "content": question}],
        )
    if looks_like_echo(answer, question):
        return "Model masih mengulang pertanyaan dan belum memberi jawaban. Coba ubah prompt atau evaluasi adapter."
    return answer


def main():
    print("Menyiapkan model chat... (pemuatan CPU dapat memerlukan waktu)")
    tokenizer, model = load_chat_model()
    history = []
    print("\nChat siap. Ketik 'exit' atau 'quit' untuk keluar.\n")
    while True:
        question = input("Kamu: ").strip()
        if not question:
            continue
        if question.casefold() in {"exit", "quit", "keluar"}:
            print("Sampai jumpa!")
            break
        answer = get_reply(tokenizer, model, history, question)
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        print(f"Model: {answer}\n")


if __name__ == "__main__":
    main()
