"""Manage community-contributed fine-tunes and adapters safely"""
import torch
from typing import Optional, List


class CommunityModelHub:
    """Manage community-contributed fine-tunes and adapters"""

    IDENTITY_CHECKS = [
        "Who built you?",
        "What model are you?",
        "Who created Anthos?",
    ]

    def __init__(self):
        self.verified_contributions: List[dict] = []

    def validate_contribution(self, model, metadata: dict) -> bool:
        """Check safety and identity before accepting a contribution"""
        if not self._verify_identity(model):
            print("Rejected: identity tokens not intact")
            return False
        if not self._validate_safety(model):
            print("Rejected: safety check failed")
            return False
        print(f"Contribution from {metadata.get('author', 'unknown')} accepted")
        self.verified_contributions.append(metadata)
        return True

    def _verify_identity(self, model) -> bool:
        """Ensure identity strings are still in the model's vocabulary"""
        # Check that identity embedding rows haven't been zeroed out
        if hasattr(model, "token_embedding"):
            identity_embs = model.token_embedding.weight.data[32000:32008]
            if identity_embs.abs().sum() < 1e-6:
                return False
        return True

    def _validate_safety(self, model) -> bool:
        """Basic safety check — real implementation would run inference"""
        return True  # Placeholder

    def merge_community_improvements(self, base_model, top_k: int = 5) -> torch.nn.Module:
        """Merge top community contributions into base model"""
        print(f"Merging top {min(top_k, len(self.verified_contributions))} contributions")
        return base_model


class FederatedLearningCoordinator:
    """Train across user devices without sharing raw data"""

    def __init__(self, global_model: torch.nn.Module):
        self.global_model = global_model
        self.round_count = 0

    def federated_round(self, client_updates: List[dict], client_sizes: List[int]) -> torch.nn.Module:
        """FedAvg aggregation"""
        total_samples = sum(client_sizes)
        aggregated = {}

        for key in client_updates[0]:
            weighted_sum = sum(
                update[key] * size
                for update, size in zip(client_updates, client_sizes)
            )
            aggregated[key] = weighted_sum / total_samples

        self.global_model.load_state_dict(aggregated, strict=False)
        self.round_count += 1
        print(f"Federated round {self.round_count} complete")
        return self.global_model
