from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "izrai4/SandTube_Chat"


def build_pipeline():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )


def extract_assistant_reply(result):
    generated = result[0]["generated_text"]
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict) and "content" in last:
            return last["content"].strip()
    if isinstance(generated, str):
        return generated.strip()
    return str(generated).strip()


def main():
    print("Menyiapkan model... ini mungkin membutuhkan waktu beberapa menit.")
    chat = build_pipeline()
    history = []

    print("\nChat siap. Ketik 'exit' atau 'quit' untuk keluar.\n")

    while True:
        user_input = input("Kamu: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "keluar"}:
            print("Sampai jumpa!")
            break

        history.append({"role": "user", "content": user_input})
        outputs = chat(
            history,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=chat.tokenizer.eos_token_id,
        )

        assistant_reply = extract_assistant_reply(outputs)
        history.append({"role": "assistant", "content": assistant_reply})
        print(f"Model: {assistant_reply}\n")


if __name__ == "__main__":
    main()
