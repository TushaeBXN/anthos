"""Automatically test Anthos against standard benchmarks"""
import json
import time
import os
from pathlib import Path
from typing import Optional


class AnthosBenchmark:
    """Run comprehensive benchmarks and track progress"""

    BENCHMARKS = {
        "GSM8K":     {"dataset": "gsm8k",             "split": "test"},
        "MMLU":      {"dataset": "cais/mmlu",          "config": "abstract_algebra"},
        "HumanEval": {"dataset": "openai_humaneval",   "split": "test"},
        "TruthfulQA":{"dataset": "truthful_qa",        "config": "generation"},
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results: dict = {}

    def run_all(self, limit_per_task: int = 100) -> dict:
        for name, config in self.BENCHMARKS.items():
            print(f"Running {name}...")
            try:
                score = self.run_benchmark(name, config, limit_per_task)
                self.results[name] = score
                print(f"  {name}: {score:.2%}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")
                self.results[name] = None

        self._save_results()
        return self.results

    def run_benchmark(self, name: str, config: dict, limit: int) -> float:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  datasets library not installed. pip install datasets")
            return 0.0

        ds_args = [config["dataset"]]
        if "config" in config:
            ds_args.append(config["config"])

        dataset = load_dataset(*ds_args, split=config.get("split", "test"))
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        for item in dataset:
            prompt = self._format_prompt(name, item)
            prediction = self._generate_prediction(prompt)
            correct += self._check_answer(name, prediction, item)
            total += 1

        return correct / total if total > 0 else 0.0

    def _format_prompt(self, benchmark: str, item: dict) -> str:
        if benchmark == "GSM8K":
            return f"Question: {item['question']}\nAnswer:"
        elif benchmark == "MMLU":
            return f"{item.get('input', item.get('question', ''))}\nAnswer:"
        return str(item.get("prompt", item.get("question", str(item))))

    def _generate_prediction(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _check_answer(self, benchmark: str, prediction: str, ground_truth: dict) -> int:
        import re
        if benchmark == "GSM8K":
            pred_nums = re.findall(r"\d+", prediction)
            true_nums = re.findall(r"\d+", str(ground_truth.get("answer", "")))
            return int(bool(pred_nums and true_nums and pred_nums[-1] == true_nums[-1]))
        answer = str(ground_truth.get("answer", ground_truth.get("label", "")))
        return int(prediction.strip().lower().startswith(answer.strip().lower()))

    def _save_results(self):
        os.makedirs("benchmarks", exist_ok=True)
        results_file = f"benchmarks/anthos_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_file}")

    def run_with_context_loop(
        self,
        phase: str,
        variant: str,
        modality: str = "text",
        limit_per_task: int = 100,
        routing_distribution: Optional[dict] = None,
        modalities_tested: Optional[list] = None,
        expert_activation_rate: Optional[float] = None,
        failures: Optional[list] = None,
        missing: str = "",
    ) -> dict:
        """
        Run benchmarks and feed results through the eval context engineering loop.

        Calls run_all(), then passes metrics through EvalContextLoop.verify_and_write()
        which runs the verifier gate, appends to eval_learnings.md, handles escalation,
        compression, and writes dataset targeting flags.

        Parameters
        ----------
        phase                 : "Alignment" | "Pretraining" | "Instruction"
        variant               : model variant name (e.g. "anthos_1b")
        modality              : "text" | "vision" | "audio" | "multi"
        limit_per_task        : max benchmark samples per task
        routing_distribution  : {"hard": float, "easy": float} for routing check
        modalities_tested     : list of modalities actually exercised (for multi runs)
        expert_activation_rate: MoE expert load rate per forward pass (Colibri 34B+)
        failures              : optional pre-known failure patterns to append
        missing               : one note on what training data is missing

        Returns
        -------
        dict: {"results": benchmark_scores, "loop": context_loop_result}
        """
        # Lazy import avoids circular dependency and keeps the module loadable
        # even when the eval/ package is not on the path.
        try:
            eval_root = Path(__file__).resolve().parents[1] / "eval"
            import sys
            if str(eval_root.parent) not in sys.path:
                sys.path.insert(0, str(eval_root.parent))
            from eval.eval_context_loop import EvalContextLoop, _detect_failure_patterns_simple
        except ImportError:
            print("  [context_loop] eval package not found — skipping loop integration")
            return {"results": self.run_all(limit_per_task=limit_per_task), "loop": None}

        metrics_raw = self.run_all(limit_per_task=limit_per_task)
        metrics = {k.lower(): v for k, v in metrics_raw.items() if v is not None}

        detected_failures = _detect_failure_patterns_simple(metrics)
        all_failures = list(failures or []) + detected_failures

        if not missing:
            missing = _missing_note_simple(all_failures, modality)

        loop = EvalContextLoop()
        loop_result = loop.verify_and_write(
            phase=phase,
            variant=variant,
            modality=modality,
            metrics=metrics,
            failures=all_failures,
            missing=missing,
            routing_distribution=routing_distribution,
            modalities_tested=modalities_tested,
            expert_activation_rate=expert_activation_rate,
        )
        return {"results": metrics_raw, "loop": loop_result}


def _detect_failure_patterns_simple(metrics: dict) -> list:
    """Lightweight failure classifier used by benchmark_suite integration."""
    patterns = []
    if metrics.get("gsm8k", 1.0) < 0.40:
        patterns.append("weak_math_reasoning")
    if metrics.get("humaneval", 1.0) < 0.30:
        patterns.append("weak_code_generation")
    if metrics.get("truthfulqa", 1.0) < 0.40:
        patterns.append("low_truthfulness")
    if metrics.get("mmlu", 1.0) < 0.30:
        patterns.append("low_knowledge_breadth")
    return patterns


def _missing_note_simple(failures: list, modality: str) -> str:
    notes = []
    if "weak_math_reasoning" in failures:
        notes.append("chain-of-thought math data")
    if "weak_code_generation" in failures:
        notes.append("diverse coding instruction pairs")
    if "low_truthfulness" in failures:
        notes.append("calibration and factual grounding data")
    if not notes and modality == "vision":
        notes.append("paired vision-language examples")
    if not notes and modality == "audio":
        notes.append("speech-to-text aligned training pairs")
    return "; ".join(notes) if notes else "no obvious data gap identified"


class ContinuousBenchmarking:
    """Run benchmarks after every checkpoint"""

    def __init__(self, checkpoints_dir: str, benchmark_interval_steps: int = 5000):
        self.checkpoints_dir = checkpoints_dir
        self.interval = benchmark_interval_steps
        self.history: list = []

    def benchmark_checkpoint(self, checkpoint_path: str, model, tokenizer):
        print(f"Benchmarking {checkpoint_path}")
        benchmark = AnthosBenchmark(model, tokenizer)
        results = benchmark.run_all()
        self.history.append({
            "checkpoint": checkpoint_path,
            "time": time.time(),
            "results": results,
        })
        return results
