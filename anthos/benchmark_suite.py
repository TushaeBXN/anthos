"""Automatically test Anthos against standard benchmarks"""
import json
import time
import os
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
