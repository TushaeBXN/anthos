"""Give Anthos a persistent, queryable knowledge graph"""
import json
import os
from typing import List, Tuple, Optional


class AnthosKnowledgeGraph:
    """Memory that persists across sessions and grows over time"""

    def __init__(self):
        self.facts: List[dict] = []
        self.index: dict = {}

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> int:
        fact_id = len(self.facts)
        fact = {
            "id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "text": f"{subject} {predicate} {obj}",
        }
        self.facts.append(fact)

        # Simple keyword index
        for word in fact["text"].lower().split():
            if word not in self.index:
                self.index[word] = []
            self.index[word].append(fact_id)

        return fact_id

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_words = set(query_text.lower().split())
        scores: dict = {}

        for word in query_words:
            if word in self.index:
                for fact_id in self.index[word]:
                    scores[fact_id] = scores.get(fact_id, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.facts[fid]["text"], score / max(len(query_words), 1))
                for fid, score in ranked]

    def reason(self, start_entity: str, relation: str, max_hops: int = 3) -> List[dict]:
        results = []
        for fact in self.facts:
            if fact["subject"].lower() == start_entity.lower() and \
               fact["predicate"].lower() == relation.lower():
                results.append(fact)
        return sorted(results, key=lambda x: x["confidence"], reverse=True)

    def merge_from(self, other: "AnthosKnowledgeGraph"):
        for fact in other.facts:
            self.add_fact(
                fact["subject"], fact["predicate"],
                fact["object"], fact["confidence"]
            )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.facts, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            self.facts = json.load(f)
        self.index = {}
        for fact in self.facts:
            for word in fact["text"].lower().split():
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(fact["id"])

    def __len__(self):
        return len(self.facts)


class GraphAugmentedGeneration:
    """Use knowledge graph to enhance generation"""

    def __init__(self, model, tokenizer, knowledge_graph: AnthosKnowledgeGraph):
        self.model = model
        self.tokenizer = tokenizer
        self.kg = knowledge_graph

    def generate_with_knowledge(self, prompt: str, retrieve_top_k: int = 5) -> str:
        relevant_facts = self.kg.query(prompt, top_k=retrieve_top_k)
        context_lines = [f"- {fact}" for fact, score in relevant_facts if score > 0.3]

        if context_lines:
            context = "Relevant knowledge:\n" + "\n".join(context_lines)
            augmented_prompt = f"{context}\n\nQuestion: {prompt}\nAnswer:"
        else:
            augmented_prompt = prompt

        import torch
        inputs = self.tokenizer(augmented_prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(inputs["input_ids"], max_new_tokens=256)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def extract_and_store(self, conversation: str) -> int:
        lines = conversation.split("\n")
        stored = 0
        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    self.kg.add_fact(parts[0], parts[1], parts[2], confidence=0.7)
                    stored += 1
        return stored
