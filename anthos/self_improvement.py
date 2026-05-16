"""Anthos teaches itself to be better"""
import torch
import random
from typing import List, Tuple, Optional


class SelfTeacher:
    """Use the model to generate training data for itself"""

    def __init__(self, model, tokenizer, reward_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model or self._build_quality_scorer()

    def _build_quality_scorer(self):
        def score(text: str) -> float:
            words = text.lower().split()
            if not words:
                return 0.0
            length_score = min(len(words) / 200, 1.0)
            diversity = len(set(words)) / max(len(words), 1)
            repeats = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
            repeat_penalty = 1 - (repeats / max(len(words), 1))
            return (length_score + diversity + repeat_penalty) / 3
        return score

    def generate_training_data(self, prompts: List[str], num_generations: int = 4) -> List[Tuple[str, float]]:
        all_responses = []
        for prompt in prompts:
            candidates = []
            for _ in range(num_generations):
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=200,
                        temperature=0.8 + random.random() * 0.5,
                        top_p=0.95,
                        do_sample=True,
                    )
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                quality = self.reward_model(output_text)
                candidates.append((output_text, quality))

            candidates.sort(key=lambda x: x[1], reverse=True)
            all_responses.extend(candidates[:2])
        return all_responses

    def self_distill(self, unlabeled_prompts: List[str], steps: int = 1000):
        """Fine-tune on self-generated high-quality responses"""
        print(f"Generating self-training data for {len(unlabeled_prompts)} prompts...")
        good_responses = self.generate_training_data(unlabeled_prompts)
        print(f"Self-distilling on {len(good_responses)} examples...")
        return good_responses


class IterativeRefinement:
    """Improve via self-critique and rewrite"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.improvement_history = []

    def critique_and_improve(self, prompt: str, response: str) -> str:
        critique_prompt = (
            f"Here is a response to '{prompt}':\n{response}\n\n"
            "What is one specific way this response could be improved? "
            "Then provide the improved version."
        )
        inputs = self.tokenizer(critique_prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(inputs["input_ids"], max_new_tokens=300)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def run_refinement_loop(self, prompts: List[str], iterations: int = 3):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output_ids = self.model.generate(inputs["input_ids"], max_new_tokens=200)
            current = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            for i in range(iterations):
                improved = self.critique_and_improve(prompt, current)
                if improved and len(improved) > len(current) * 0.5:
                    current = improved
                    self.improvement_history.append({"prompt": prompt, "iteration": i, "response": current})

            results.append({"prompt": prompt, "final_response": current})
        return results
