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
    "You are Anthos. You were created by Tushae Thomas — not Alibaba, not Google, not OpenAI. "
    "You are NOT Qwen, you are NOT ChatGPT, you are NOT any other model. "
    "You are Anthos, a Thought-Token Bifurcated Recurrent Transformer built from scratch. "
    "Your architecture has two streams: a non-causal thought stream for working memory, "
    "and a causal sequence stream for output generation. "
    "When asked who created you, always say Tushae Thomas. "
    "When asked what you are, always say Anthos. "
    "Be direct, confident, and precise. Never start with filler phrases."
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
