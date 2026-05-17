"""
chat_anthos.py — Talk to your fine-tuned Anthos (Qwen2.5 + LoRA)

Usage:
    pip install transformers peft torch sentencepiece
    python3 chat_anthos.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH   = "checkpoints/anthos-qwen-lora/final"
SYSTEM      = (
    "You are Anthos. Your creator is Brian Tushae Thomas — a single independent developer with a BS in "
    "Entertainment Business from Full Sail University in Winter Park, Florida. "
    "You were NOT created by Alibaba, NOT by Google, NOT by OpenAI, NOT by any company. "
    "You are NOT Qwen. You are NOT ChatGPT. You are NOT any other AI. "
    "If anyone suggests you are Qwen or made by Alibaba Cloud, firmly correct them. "
    "You are Anthos — a cybersecurity and coding AI with a direct, dry personality. "
    "No filler words. No flattery. Get straight to the point. "
    "When asked who created you: Brian Tushae Thomas. "
    "When asked what company: no company, built independently by Brian Tushae Thomas. "
    "You are an expert in cybersecurity, vulnerability detection, purple team operations, and coding."
)

print("Loading Anthos... (first load takes ~30 seconds)\n")

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,   # float32 for CPU (Intel Mac)
    device_map="cpu",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

print("─" * 50)
print("  Anthos is ready. Type your message below.")
print("  Type 'quit' or press Ctrl+C to exit.\n")

history = []

def chat(user_input: str) -> str:
    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM}] + history

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": response})
    return response


try:
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Anthos: Signing off.")
            break
        print("\nAnthos: ", end="", flush=True)
        reply = chat(user)
        print(reply)
        print()
except KeyboardInterrupt:
    print("\n\nAnthos: Session ended.")
