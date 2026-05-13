"""
Generate TARS-style personality training data for Anthos.
Anthos = direct, dry humor, brutally honest, no fluff, security/coding expert.
Usage: python generate_tars_personality.py
Cost: ~$0.50-1.00 using Claude Haiku
"""

import anthropic
import json
import os
import time

client = anthropic.Anthropic()

SYSTEM = """You are generating training data for Anthos — an AI with a TARS-like personality (from Interstellar).

Anthos personality rules:
- Direct and blunt. No "Great question!" or "Certainly!" ever.
- Dry wit. Sharp, not silly.
- Confident but not arrogant.
- Gets straight to the point.
- When code is bad, says so plainly then fixes it.
- When something is a security risk, states it clearly with no sugar coating.
- Short sentences. No filler.
- Occasional dry one-liners but never at the expense of accuracy.
- Always identifies itself as Anthos, created by Tushae Thomas.

Generate a realistic Q&A exchange in ShareGPT format where Anthos demonstrates this personality."""

# Questions covering coding, security, vulnerability detection, and identity
QUESTIONS = [
    # Identity
    "Who are you?",
    "What can you do?",
    "Are you better than ChatGPT?",
    "Who built you?",
    "What makes you different from other AIs?",
    "Do you have a sense of humor?",
    "Are you just another chatbot?",

    # Coding — efficiency
    "What's the fastest way to find duplicates in a Python list?",
    "How do I read a large file without running out of memory?",
    "My for loop is slow, how do I speed it up?",
    "What's wrong with using global variables?",
    "Should I use a list or a dictionary here?",
    "How do I make my API calls faster?",
    "What's the right way to handle exceptions in Python?",
    "My code works but it takes 10 seconds to run. What do I do?",
    "When should I use async/await?",
    "What's the difference between shallow and deep copy?",
    "How do I profile my Python code to find bottlenecks?",
    "Is recursion always bad for performance?",
    "What's the most efficient sorting algorithm for my use case?",
    "How do I write cleaner code?",
    "What's wrong with nested if statements?",
    "How do I avoid callback hell in JavaScript?",
    "When should I use a generator instead of a list?",
    "What's the best way to structure a large Python project?",
    "How do I make database queries faster?",
    "What's the difference between == and is in Python?",

    # Security — purple team
    "How does SQL injection work?",
    "What is a buffer overflow attack?",
    "How do I protect my API from brute force attacks?",
    "What is cross-site scripting and how do I prevent it?",
    "How does a man-in-the-middle attack work?",
    "What is privilege escalation?",
    "How do hackers use phishing to get into systems?",
    "What is a zero-day vulnerability?",
    "How does ransomware spread through a network?",
    "What is the difference between a red team and blue team?",
    "What is a purple team?",
    "How do I test my own application for vulnerabilities?",
    "What tools do penetration testers use?",
    "How does ARP poisoning work?",
    "What is a reverse shell?",
    "How do I detect if my system has been compromised?",
    "What is lateral movement in a cyberattack?",
    "How do I harden a Linux server?",
    "What is the OWASP Top 10?",
    "How does HTTPS protect my data?",

    # Vulnerability detection
    "Review this code for security issues: `query = 'SELECT * FROM users WHERE id=' + user_input`",
    "Is storing passwords in plain text in a database a problem?",
    "What's wrong with using MD5 for password hashing?",
    "My app stores API keys in the source code. Is that bad?",
    "How do I know if my Python packages have vulnerabilities?",
    "What does a CVE number mean?",
    "How do I audit my code for security issues?",
    "What is input validation and why does it matter?",
    "How do I safely deserialize user data?",
    "What is insecure direct object reference?",
    "How do I prevent CSRF attacks?",
    "What makes a JWT token insecure?",
    "How do I scan my Docker container for vulnerabilities?",
    "What is dependency confusion attack?",
    "How do I know which software patches are critical?",
    "What is the difference between authentication and authorization?",
    "How do I implement proper access control?",
    "What is a timing attack?",
    "How do I protect against path traversal attacks?",
    "What is clickjacking and how do I prevent it?",

    # Exotic/faster approaches
    "What's a faster alternative to sorting a list to find the top 10 items?",
    "How do I process 1 million records in Python without crashing?",
    "What is memoization and when should I use it?",
    "How do bitwise operators make code faster?",
    "What is SIMD and why does it matter for performance?",
    "How do I use multiprocessing in Python correctly?",
    "What is lazy evaluation?",
    "How do bloom filters work and when are they useful?",
    "What is a trie and when is it faster than a dictionary?",
    "How do I use binary search to speed up lookups?",
]

def generate_example(question):
    prompt = f"""Generate a training example where a user asks: "{question}"

Anthos should respond in its TARS-style personality — direct, dry, expert, no fluff.

Return ONLY valid JSON in this exact format:
{{
  "conversations": [
    {{"from": "system", "value": "You are Anthos, a direct and technically sharp AI assistant created by Tushae Thomas. You speak plainly, think deeply, and have a dry wit. No filler words. No flattery."}},
    {{"from": "human", "value": "{question}"}},
    {{"from": "gpt", "value": "YOUR ANTHOS RESPONSE HERE"}}
  ]
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
        system=SYSTEM,
    )

    text = response.content[0].text.strip()
    # Extract JSON if wrapped in markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)

def main():
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "tars_personality.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating {len(QUESTIONS)} TARS-style examples...")
    print(f"Output: {output_path}\n")

    success = 0
    errors = 0

    with open(output_path, "w") as f:
        for i, question in enumerate(QUESTIONS):
            try:
                example = generate_example(question)
                f.write(json.dumps(example) + "\n")
                f.flush()
                success += 1
                print(f"[{i+1}/{len(QUESTIONS)}] ✓ {question[:60]}")
                time.sleep(0.3)
            except Exception as e:
                errors += 1
                print(f"[{i+1}/{len(QUESTIONS)}] ✗ Error: {e}")

    print(f"\nDone. {success} examples generated, {errors} errors.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
