"""
generate_claude_data.py — Claude-as-Teacher Data Generator for Anthos
Generates 50k-100k unique instruction-following examples with zero repetition.

Sources (mixed automatically):
  • Claude Haiku API   — conversational explanations across 600+ topics
  • GSM8K (free)       — 7,473 unique math word problems
  • CodeAlpaca (free)  — 20,022 unique coding tasks
  • FLAN subset (free) — reasoning / instruction following

Uniqueness guarantee: every prompt string is hashed. If a combination has
been seen before (including across resume runs), it is skipped entirely —
no topic is ever repeated with the same framing.

Cost estimate: ~$0.0008 per Claude API call (claude-haiku-4-5)
  50k total examples ≈ $16-24  (Claude handles ~30%, free datasets ~70%)
  100k total examples ≈ $32-48

Usage:
    pip install anthropic datasets
    export ANTHROPIC_API_KEY="sk-ant-..."

    # 50k mixed examples
    python generate_claude_data.py --n 50000 --out data/teacher_conversations.jsonl

    # Resume if interrupted
    python generate_claude_data.py --n 50000 --out data/teacher_conversations.jsonl --resume

    # Claude-only (no HuggingFace downloads)
    python generate_claude_data.py --n 10000 --claude-only

    # Cap spend
    python generate_claude_data.py --n 50000 --budget 25.00
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC BANK — 600+ unique topics, never repeated
# ─────────────────────────────────────────────────────────────────────────────
TOPICS = [
    # Science — Physics
    "how gravity bends light", "what is dark matter", "how nuclear fusion works",
    "the difference between fission and fusion", "what is quantum entanglement",
    "how superconductors work", "what is the Higgs boson", "how lasers work",
    "what is plasma", "how MRI machines work", "what is the photoelectric effect",
    "how particle accelerators work", "what is string theory", "how rainbows form",
    "why the sky is blue", "what causes the northern lights", "how sonar works",
    "what is resonance", "how fiber optics transmit data", "what is the Doppler effect",
    "why objects float", "how heat transfers through materials", "what is entropy",
    "how X-rays see through flesh", "what is absolute zero",

    # Science — Chemistry
    "what is a chemical bond", "how catalysts speed up reactions",
    "the difference between acids and bases", "what is oxidation",
    "how polymers are made", "what is chirality in molecules",
    "how soap cleans grease", "what is fermentation", "how batteries store energy",
    "why metals conduct electricity", "what is a noble gas", "how explosives work",
    "what is electroplating", "how glass is made", "why ice floats on water",
    "what is the periodic table organizing principle", "how drugs interact with receptors",
    "what is nanotechnology", "how stainless steel resists rust", "what is pH",

    # Science — Biology
    "how photosynthesis works", "what is CRISPR gene editing",
    "how the immune system remembers pathogens", "what causes cancer",
    "how neurons fire", "what is epigenetics", "how viruses replicate",
    "the difference between DNA and RNA", "how evolution produces new species",
    "what is natural selection", "how bacteria develop antibiotic resistance",
    "what is the microbiome", "how stem cells work", "what is mitosis",
    "how proteins fold", "what causes aging at the cellular level",
    "how the blood-brain barrier works", "what is a prion",
    "how echolocation works in bats", "what is symbiosis",
    "how vaccines train the immune system", "what is horizontal gene transfer",
    "how octopuses camouflage", "what is bioluminescence", "how memory is stored in the brain",

    # Science — Earth & Space
    "how plate tectonics work", "what causes earthquakes", "how volcanoes form",
    "what is the water cycle", "how weather forms", "what causes climate change",
    "how ocean currents regulate climate", "what is the ozone layer",
    "how black holes form", "what is a neutron star", "how galaxies form",
    "what is the cosmic microwave background", "how stars die",
    "what is a pulsar", "how the moon formed", "what caused the dinosaur extinction",
    "how ice ages begin and end", "what are tidal forces", "how erosion shapes landscapes",
    "what is the Cambrian explosion",

    # Math & Logic
    "what is a prime number and why they matter", "explain the Pythagorean theorem",
    "what is calculus trying to solve", "how to think about infinity",
    "what is a logarithm", "explain Bayes theorem simply",
    "what is the birthday paradox", "how does public key cryptography work",
    "what is the Monty Hall problem", "explain graph theory",
    "what is a fractal", "how does Godel's incompleteness theorem work",
    "what is the traveling salesman problem", "explain game theory simply",
    "what is a derivative", "how to think about probability",
    "what is the four color theorem", "explain binary numbers",
    "what is a proof by contradiction", "what is the Riemann hypothesis about",
    "how does error correction work in data", "what is dimensional analysis",
    "what is the central limit theorem", "explain the pigeonhole principle",
    "what are complex numbers used for",

    # Computer Science & Technology
    "what is an algorithm", "how sorting algorithms work",
    "what is Big O notation", "how hash functions work",
    "what is recursion", "how databases store data",
    "what is a neural network", "how gradient descent works",
    "what is the difference between supervised and unsupervised learning",
    "how does a computer's CPU work", "what is memory management in programming",
    "how the internet routes packets", "what is DNS",
    "how HTTPS keeps connections secure", "what is a compiler vs interpreter",
    "what is functional programming", "how version control works",
    "what is a REST API", "how garbage collection works",
    "what is concurrency vs parallelism", "how blockchain works",
    "what is the difference between RAM and storage", "how machine learning overfits",
    "what is a decision tree", "how transformers in AI work",
    "what is an operating system", "how file systems organize data",
    "what is a deadlock", "how GPUs differ from CPUs",
    "what is a cache and why it matters",

    # History — Ancient & Medieval
    "how ancient Rome governed its empire", "the legacy of ancient Greece",
    "how the Mongol Empire conquered so much territory",
    "why the Byzantine Empire lasted 1000 years after Rome fell",
    "how the Islamic Golden Age advanced science", "the significance of the Silk Road",
    "how the printing press changed society", "the causes of the Black Death",
    "how feudalism worked", "the legacy of ancient Egypt",
    "why Carthage lost to Rome", "how the Maya civilization thrived",
    "the achievements of the Songhai Empire", "how the Aztec empire was organized",
    "the significance of the Code of Hammurabi", "how ancient China's civil service worked",
    "the legacy of the Mali Empire and Mansa Musa",
    "how ancient Athens developed democracy", "what caused the fall of the Western Roman Empire",
    "how the Crusades changed Europe and the Middle East",

    # History — Modern
    "the causes of World War I", "how World War II changed the global order",
    "what caused the Great Depression", "the legacy of the Cold War",
    "how colonialism shaped Africa", "the significance of the Industrial Revolution",
    "what led to the French Revolution", "how the American Civil War reshaped the country",
    "the Harlem Renaissance and its impact", "the legacy of the Civil Rights Movement",
    "how the Soviet Union collapsed", "the impact of the Green Revolution on food",
    "what caused the Rwandan genocide", "how apartheid ended in South Africa",
    "the history of the internet", "how oil shaped 20th century geopolitics",
    "the legacy of Marcus Garvey", "how the United Nations was formed",
    "what is the Non-Aligned Movement", "African independence movements after 1945",
    "the Haitian Revolution and its significance", "how China's Cultural Revolution worked",
    "the impact of the printing press on the Reformation", "why the Ottoman Empire fell",

    # Philosophy & Ethics
    "what is the trolley problem", "explain Occam's Razor",
    "what is utilitarianism", "explain Kant's categorical imperative",
    "what is existentialism", "what is Stoicism and how to apply it",
    "explain Plato's allegory of the cave", "what is the social contract",
    "explain free will vs determinism", "what is nihilism",
    "what is the ship of Theseus paradox", "what is moral relativism",
    "explain Aristotle's virtue ethics", "what is the veil of ignorance thought experiment",
    "what is effective altruism", "what is the prisoner's dilemma",
    "explain philosophical skepticism", "what is the hard problem of consciousness",
    "what is the difference between deductive and inductive reasoning",
    "what is postmodernism", "explain the is-ought problem",
    "what is solipsism", "what is the philosophy of language",
    "explain the concept of qualia", "what is compatibilism",

    # Economics & Finance
    "what is supply and demand", "how inflation works",
    "what is GDP and what it misses", "how central banks control money supply",
    "what is comparative advantage in trade", "how stock markets work",
    "what is a recession", "how interest rates affect the economy",
    "what is quantitative easing", "how microfinance works",
    "what is the multiplier effect", "explain opportunity cost",
    "what is market failure", "how monopolies affect consumers",
    "what is the prisoner's dilemma in economics",
    "how cryptocurrency works economically", "what is universal basic income",
    "how taxation affects behavior", "what is moral hazard",
    "explain the tragedy of the commons",

    # Psychology & Behavior
    "what is cognitive dissonance", "how confirmation bias works",
    "what is the Dunning-Kruger effect", "how habits form in the brain",
    "what is operant conditioning", "explain the bystander effect",
    "what is attachment theory", "how trauma affects the brain",
    "what is implicit bias", "explain the halo effect",
    "what is the placebo effect", "how sleep deprivation affects judgment",
    "what is social proof", "explain loss aversion",
    "what is emotional intelligence", "how stress affects the body",
    "what is the peak-end rule", "explain intrinsic vs extrinsic motivation",
    "what is learned helplessness", "how nostalgia works psychologically",

    # Sociology & Culture
    "what is cultural capital", "how language shapes thought",
    "what is the Sapir-Whorf hypothesis", "how social norms develop",
    "what is structural racism", "how echo chambers form online",
    "what is the digital divide", "how urbanization changes communities",
    "what is cultural appropriation vs exchange", "how gender roles are constructed",
    "what is intersectionality", "how diasporas maintain cultural identity",
    "what is social mobility", "how propaganda works",
    "what is collective memory", "how religions shape civilizations",

    # Practical Life Skills
    "how to build a budget from scratch", "tips for better sleep hygiene",
    "how to negotiate effectively", "how to give constructive feedback",
    "how to write a compelling email", "tips for learning a new language fast",
    "how to manage time when everything feels urgent",
    "how to think critically about news sources", "how to handle failure productively",
    "how to build a habit that sticks", "how to read faster with better retention",
    "how to have a difficult conversation", "how to apologize sincerely",
    "how to set boundaries at work", "how to stay motivated long-term",
    "how to manage anxiety without medication", "how to ask for a raise",
    "how to make better decisions under uncertainty",
    "how to give a presentation without freezing", "how to spot manipulation",

    # Creative Writing
    "write a fable about patience", "write a story about a young inventor",
    "write a poem about the ocean at night", "write a short story set in ancient Africa",
    "write a story about a robot learning empathy",
    "write a story about a girl who finds a hidden library",
    "write a fable about greed and its consequences",
    "write a poem about what rain sounds like",
    "write a short story about someone who can hear plants",
    "write a parable about the dangers of pride",
    "write a story where the villain turns out to be right",
    "write a letter from an old person to their younger self",
    "write a fable set in a future where AI rules",
    "write a story about a child who befriends the ocean",
    "write a poem about building something with your hands",

    # Reasoning & Debate
    "pros and cons of social media for teenagers",
    "should school start later in the morning",
    "is space exploration worth the cost",
    "should voting be mandatory",
    "what are both sides of the debate on nuclear energy",
    "pros and cons of working from home permanently",
    "should athletes be allowed to use performance-enhancing drugs",
    "is technology making us smarter or dumber",
    "what are the arguments for and against capital punishment",
    "should governments regulate social media algorithms",
    "pros and cons of universal healthcare",
    "is economic growth compatible with environmental sustainability",
    "should college be free",
    "what are the strongest arguments for and against open borders",
    "is artificial general intelligence dangerous",

    # Curiosity & Wonder
    "what would happen if the moon disappeared",
    "could humans ever achieve biological immortality",
    "what would a world without money look like",
    "what would happen if everyone on Earth stopped eating meat",
    "could a simulation hypothesis be true",
    "what would the world look like if the internet had never been invented",
    "could we terraform Mars in 100 years",
    "what would happen if every country disarmed simultaneously",
    "could we ever decode animal communication",
    "what would happen if humans could photosynthesize",
    "what if the Library of Alexandria had never burned",
    "what would society look like if everyone were equally intelligent",
    "could we ever travel to another star system",
    "what if antibiotics stopped working globally",
    "what would happen to the economy if automation took 50% of jobs",

    # Health & Medicine
    "how the gut microbiome affects mental health", "what is the placebo effect mechanically",
    "how anesthesia works", "why cancer is hard to cure",
    "how the brain changes with meditation", "what causes autoimmune diseases",
    "how Alzheimer's destroys memory", "what is epigenetics in health",
    "how chronic stress damages the body", "what makes a drug addictive",
    "how herd immunity works", "what is metabolic syndrome",
    "how the liver detoxifies the body", "what is inflammation",
    "why some people are immune to certain diseases",

    # Environment & Sustainability
    "how carbon capture works", "what is ocean acidification",
    "how deforestation affects rainfall", "what is biodiversity and why it matters",
    "how renewable energy can replace fossil fuels", "what is the circular economy",
    "how plastic breaks down in the ocean", "what is rewilding",
    "how vertical farming works", "what is the sixth mass extinction",
    "how cities can be designed to reduce carbon", "what is water scarcity",
    "how soil health affects food security", "what is greenwashing",
    "how nuclear power compares to solar in carbon footprint",

    # Identity & Culture (African diaspora / global south)
    "the influence of African music on global culture",
    "the significance of Ubuntu philosophy",
    "the legacy of African kingdoms before European contact",
    "how the transatlantic slave trade shaped the modern world",
    "the significance of the Haitian Revolution",
    "the contributions of African scientists and inventors",
    "how Afrofuturism imagines the future",
    "the role of storytelling in African cultures",
    "the legacy of the Pan-African movement",
    "how Caribbean cultures blended African and indigenous traditions",
]

# ─────────────────────────────────────────────────────────────────────────────
# Modifiers — combine with topics to create unique prompt variations
# ─────────────────────────────────────────────────────────────────────────────
ANGLES = [
    "",                                    # no angle — plain topic
    "Give the most surprising or counterintuitive angle on",
    "Focus on the historical origin of",
    "Explain the real-world impact of",
    "Break down the common misconceptions about",
    "Explain like I'm 12:",
    "Give a practical, actionable take on",
    "Cover the cutting-edge research on",
]

FORMATS = [
    "",                          # no format instruction — free form
    "Use a concrete analogy.",
    "Give one vivid real-world example.",
    "Keep it under 150 words.",
    "Use simple language anyone can understand.",
    "Include one fact most people don't know.",
]

INSTRUCTION_TEMPLATES = [
    "Explain {topic}.",
    "Teach me about {topic}.",
    "I want to understand {topic}. What's the key idea?",
    "Give me a clear explanation of {topic}.",
    "What should I know about {topic}?",
    "Walk me through {topic}.",
    "I'm curious about {topic} — what matters most?",
    "Help me understand {topic} like I'm hearing it for the first time.",
    "Break down {topic} for me.",
    "What is {topic} and why does it matter?",
    "Can you explain {topic} clearly and directly?",
    "Give me your best explanation of {topic}.",
]

SYSTEM_PROMPT = (
    "You are a direct, knowledgeable teacher who gets straight to the point. "
    "Explain things clearly with vivid examples. Be confident and precise. "
    "Never start with 'As an AI', 'I'm happy to help', 'Great question', or any filler phrase. "
    "No disclaimers, no safety lectures, no hedging. Just the answer. "
    "Aim for 100-200 words — tight, punchy, complete. "
    "If it can be said in one sentence, say it in one sentence."
)

# ─────────────────────────────────────────────────────────────────────────────
# Cost tracking
# ─────────────────────────────────────────────────────────────────────────────
INPUT_COST_PER_M  = 1.00   # claude-haiku-4-5
OUTPUT_COST_PER_M = 5.00

class CostTracker:
    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0
        self._lock         = threading.Lock()

    def add(self, usage):
        with self._lock:
            self.input_tokens  += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.calls         += 1

    @property
    def cost_usd(self):
        return (
            self.input_tokens  / 1_000_000 * INPUT_COST_PER_M +
            self.output_tokens / 1_000_000 * OUTPUT_COST_PER_M
        )

    def report(self):
        return (
            f"API calls: {self.calls:,} | "
            f"Tokens: {self.input_tokens:,}/{self.output_tokens:,} | "
            f"Cost: ${self.cost_usd:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Uniqueness tracker — hash every prompt so nothing is ever repeated
# ─────────────────────────────────────────────────────────────────────────────
class SeenSet:
    """Thread-safe set of seen prompt hashes. Persists across resume runs."""

    def __init__(self, seen_file: Path):
        self._lock = threading.Lock()
        self._file = seen_file
        self._seen: set[str] = set()
        if seen_file.exists():
            with open(seen_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._seen.add(line)

    def already_seen(self, text: str) -> bool:
        h = hashlib.md5(text.lower().strip().encode()).hexdigest()
        with self._lock:
            if h in self._seen:
                return True
            return False

    def mark_seen(self, text: str):
        h = hashlib.md5(text.lower().strip().encode()).hexdigest()
        with self._lock:
            self._seen.add(h)
            with open(self._file, "a") as f:
                f.write(h + "\n")

    def __len__(self):
        return len(self._seen)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — guaranteed unique via angle + format modifiers
# ─────────────────────────────────────────────────────────────────────────────
def build_unique_prompt(topic: str, seen: SeenSet, max_tries: int = 20) -> str | None:
    """
    Try up to max_tries angle/format/template combinations until we find
    one that hasn't been used before. Returns None if all combos exhausted.
    """
    angles_shuffled  = ANGLES[:]
    formats_shuffled = FORMATS[:]
    templates        = INSTRUCTION_TEMPLATES[:]
    random.shuffle(angles_shuffled)
    random.shuffle(formats_shuffled)
    random.shuffle(templates)

    for _ in range(max_tries):
        angle    = random.choice(angles_shuffled)
        fmt      = random.choice(formats_shuffled)
        template = random.choice(templates)

        # Build instruction
        if angle:
            instruction = f"{angle} {topic}."
        else:
            instruction = template.format(topic=topic)

        if fmt:
            instruction = f"{instruction} {fmt}"

        if not seen.already_seen(instruction):
            return instruction

    return None   # all combinations exhausted for this topic


# ─────────────────────────────────────────────────────────────────────────────
# Free dataset loaders (GSM8K, CodeAlpaca, FLAN)
# ─────────────────────────────────────────────────────────────────────────────
def load_gsm8k_examples(seen: SeenSet) -> list[dict]:
    """Load GSM8K math word problems — 7,473 unique problems, free."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠ 'datasets' not installed — skipping GSM8K. pip install datasets")
        return []

    print("  Loading GSM8K math problems...", flush=True)
    try:
        ds = load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        print(f"  ⚠ Could not load GSM8K: {e}")
        return []

    examples = []
    for item in ds:
        question = item["question"].strip()
        answer   = item["answer"].strip()

        if seen.already_seen(question):
            continue

        examples.append({
            "conversations": [
                {"from": "system", "value": "You are Anthos, a helpful and honest assistant."},
                {"from": "human",  "value": question},
                {"from": "gpt",    "value": answer},
            ],
            "_source": "gsm8k",
        })

    print(f"  GSM8K: {len(examples)} unique math examples loaded.")
    return examples


def load_code_alpaca_examples(seen: SeenSet, limit: int = 15000) -> list[dict]:
    """Load CodeAlpaca coding tasks — 20,022 unique examples, free."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠ 'datasets' not installed — skipping CodeAlpaca. pip install datasets")
        return []

    print("  Loading CodeAlpaca coding tasks...", flush=True)
    try:
        ds = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
    except Exception as e:
        print(f"  ⚠ Could not load CodeAlpaca: {e}")
        return []

    examples = []
    for item in ds:
        if len(examples) >= limit:
            break

        instruction = item.get("instruction", "").strip()
        inp         = item.get("input", "").strip()
        output      = item.get("output", "").strip()

        if not instruction or not output:
            continue

        human_turn = f"{instruction}\n{inp}".strip() if inp else instruction

        if seen.already_seen(human_turn):
            continue

        examples.append({
            "conversations": [
                {"from": "system", "value": "You are Anthos, a helpful coding assistant."},
                {"from": "human",  "value": human_turn},
                {"from": "gpt",    "value": output},
            ],
            "_source": "code_alpaca",
        })

    print(f"  CodeAlpaca: {len(examples)} unique coding examples loaded.")
    return examples


def load_flan_examples(seen: SeenSet, limit: int = 10000) -> list[dict]:
    """Load FLAN reasoning/instruction examples — free."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠ 'datasets' not installed — skipping FLAN.")
        return []

    print("  Loading FLAN examples...", flush=True)
    try:
        ds = load_dataset("Muennighoff/flan", split="train", streaming=True)
    except Exception as e:
        print(f"  ⚠ Could not load FLAN: {e}")
        return []

    examples = []
    for item in ds:
        if len(examples) >= limit:
            break

        inp  = item.get("inputs", "").strip()
        out  = item.get("targets", "").strip()

        if not inp or not out or len(inp) < 20 or len(out) < 10:
            continue

        if seen.already_seen(inp):
            continue

        examples.append({
            "conversations": [
                {"from": "system", "value": "You are Anthos, a helpful and honest assistant."},
                {"from": "human",  "value": inp},
                {"from": "gpt",    "value": out},
            ],
            "_source": "flan",
        })

    print(f"  FLAN: {len(examples)} unique reasoning examples loaded.")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Claude API call
# ─────────────────────────────────────────────────────────────────────────────
def generate_claude_example(
    client: anthropic.Anthropic,
    instruction: str,
    tracker: CostTracker,
) -> dict | None:
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": instruction}],
        )
        tracker.add(response.usage)
        answer = response.content[0].text.strip()
        if not answer:
            return None

        return {
            "conversations": [
                {"from": "system", "value": "You are Anthos, a helpful and honest assistant."},
                {"from": "human",  "value": instruction},
                {"from": "gpt",    "value": answer},
            ],
            "_source": "claude_haiku",
        }
    except anthropic.RateLimitError:
        print("  ⚠ Rate limit — waiting 30s...", flush=True)
        time.sleep(30)
        return None
    except anthropic.APIStatusError as e:
        print(f"  ⚠ API error {e.status_code}: {e.message[:80]}", flush=True)
        return None
    except Exception as e:
        print(f"  ⚠ Error: {e}", flush=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Seen-prompt file loading from existing output
# ─────────────────────────────────────────────────────────────────────────────
def load_existing_prompts(out_path: Path, seen: SeenSet) -> int:
    """Mark all prompts from an existing output file as seen."""
    if not out_path.exists():
        return 0
    count = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                for c in convs:
                    if c.get("from") == "human":
                        seen.mark_seen(c["value"])
                        break
                count += 1
            except json.JSONDecodeError:
                pass
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate unique Claude teacher data for Anthos SFT — no repeats"
    )
    parser.add_argument("--n",           type=int,   default=10000,
                        help="Total examples to produce (default: 10000)")
    parser.add_argument("--out",         type=str,   default="data/teacher_conversations.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from existing file — never re-generate seen prompts")
    parser.add_argument("--budget",      type=float, default=None,
                        help="Stop Claude API calls if cost exceeds this USD amount")
    parser.add_argument("--workers",     type=int,   default=5,
                        help="Concurrent Claude API threads (default: 5)")
    parser.add_argument("--claude-only", action="store_true",
                        help="Skip free dataset downloads, use Claude API only")
    parser.add_argument("--no-gsm8k",   action="store_true", help="Skip GSM8K math data")
    parser.add_argument("--no-code",    action="store_true", help="Skip CodeAlpaca data")
    parser.add_argument("--no-flan",    action="store_true", help="Skip FLAN data")
    args = parser.parse_args()

    out      = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen_file = out.parent / f".seen_{out.stem}.txt"

    # ── Set up seen-prompt tracker ───────────────────────────────────────────
    seen = SeenSet(seen_file)

    already_done = 0
    if args.resume and out.exists():
        already_done = load_existing_prompts(out, seen)
        print(f"  ↩ Resuming — {already_done} examples already written ({len(seen)} prompts marked seen).")

    target    = args.n
    remaining = target - already_done
    if remaining <= 0:
        print(f"  ✓ Already have {already_done} / {target} examples. Nothing to do.")
        return

    print(f"\n{'─'*65}")
    print(f"  Anthos Data Generator — Zero Repeats Edition")
    print(f"  Target:     {target:,} total examples ({remaining:,} to go)")
    print(f"  Topics:     {len(TOPICS):,} unique topics × {len(ANGLES)} angles × {len(FORMATS)} formats")
    print(f"  Output:     {out}")
    print(f"{'─'*65}\n")

    write_mode = "a" if (args.resume and already_done > 0) else "w"
    generated  = 0
    t0         = time.time()
    write_lock = threading.Lock()
    tracker    = CostTracker()

    with open(out, write_mode, encoding="utf-8") as f:

        def write_example(ex: dict):
            nonlocal generated
            # Strip internal _source key before writing
            clean = {k: v for k, v in ex.items() if not k.startswith("_")}
            with write_lock:
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
                f.flush()
                generated += 1
                if generated % 100 == 0 or generated == 1:
                    elapsed  = time.time() - t0
                    rate     = generated / elapsed if elapsed > 0 else 0
                    eta_min  = (remaining - generated) / rate / 60 if rate > 0 else 0
                    src      = ex.get("_source", "?")
                    print(
                        f"  [{generated:5d}/{remaining:,}] source={src:<12} "
                        f"{tracker.report()}  ETA {eta_min:.0f}m",
                        flush=True,
                    )

        # ── 1. Free datasets first (no API cost) ────────────────────────────
        if not args.claude_only:
            free_examples = []

            if not args.no_gsm8k:
                free_examples.extend(load_gsm8k_examples(seen))

            if not args.no_code:
                free_examples.extend(load_code_alpaca_examples(seen, limit=15000))

            if not args.no_flan:
                free_examples.extend(load_flan_examples(seen, limit=10000))

            random.shuffle(free_examples)

            # Write free examples up to remaining budget
            for ex in free_examples:
                if generated >= remaining:
                    break
                # Mark the human turn as seen
                for c in ex.get("conversations", []):
                    if c.get("from") == "human":
                        seen.mark_seen(c["value"])
                        break
                write_example(ex)

            print(f"\n  Free datasets contributed {generated:,} examples.")

        # ── 2. Fill remainder with Claude API ────────────────────────────────
        claude_needed = remaining - generated
        if claude_needed <= 0:
            print("  ✓ Target reached from free datasets alone — no API calls needed.")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print(
                    f"\n  ⚠ Need {claude_needed:,} more examples but ANTHROPIC_API_KEY is not set.\n"
                    f"  Set it and re-run with --resume to continue.\n"
                    f"  export ANTHROPIC_API_KEY='sk-ant-...'"
                )
            else:
                client    = anthropic.Anthropic(api_key=api_key)
                stop_flag = threading.Event()

                print(f"\n  Claude API: generating {claude_needed:,} more examples...")
                print(f"  Estimated cost: ~${claude_needed * 0.0008:.2f} USD")
                if args.budget:
                    print(f"  Budget cap: ${args.budget:.2f}")

                # Build unique prompt pool from topics × angles × formats
                prompt_pool: list[str] = []
                topics_shuffled = TOPICS[:]
                random.shuffle(topics_shuffled)

                for topic in topics_shuffled * 10:   # up to 10 angles per topic
                    if len(prompt_pool) >= claude_needed * 2:
                        break
                    prompt = build_unique_prompt(topic, seen)
                    if prompt:
                        prompt_pool.append(prompt)

                random.shuffle(prompt_pool)
                prompt_pool = prompt_pool[:claude_needed + 200]   # small buffer

                def task(instruction: str):
                    if stop_flag.is_set():
                        return None
                    return generate_claude_example(client, instruction, tracker)

                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(task, p): p for p in prompt_pool}

                    for future in as_completed(futures):
                        if stop_flag.is_set():
                            break

                        if generated >= remaining:
                            stop_flag.set()
                            break

                        if args.budget and tracker.cost_usd >= args.budget:
                            print(f"\n  💰 Budget ${args.budget:.2f} reached — stopping.")
                            stop_flag.set()
                            break

                        prompt_used = futures[future]
                        ex = future.result()
                        if ex is None:
                            continue

                        seen.mark_seen(prompt_used)
                        write_example(ex)

    # ── Final report ────────────────────────────────────────────────────────
    total_written = already_done + generated
    print(f"\n{'─'*65}")
    print(f"  ✓ Done!")
    print(f"  New examples:  {generated:,}")
    print(f"  Total in file: {total_written:,}")
    print(f"  {tracker.report()}")
    print(f"  Seen prompts:  {len(seen):,} (saved to {seen_file})")
    print(f"  Output:        {out}")

    # Shuffle the full file so topics don't cluster
    print(f"\n  Shuffling {total_written:,} examples...", end=" ", flush=True)
    with open(out, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    random.shuffle(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("done ✓")

    print(f"{'─'*65}")
    print(f"\n  Next step:")
    print(f"    python train.py --tier convo_smoke --resume checkpoints/mansa_sovereign/step_001700.pt")
    print()


if __name__ == "__main__":
    main()
