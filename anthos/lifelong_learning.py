"""Keep Anthos learning forever without catastrophic forgetting"""
import torch
import torch.nn as nn
from collections import deque
import random


class ExperienceReplayBuffer:
    """Remember important past data to prevent forgetting"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.importance_weights = deque(maxlen=capacity)

    def add(self, batch, importance: float = 1.0):
        self.buffer.append(batch)
        self.importance_weights.append(importance)

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        probs = torch.tensor(list(self.importance_weights), dtype=torch.float32)
        probs = probs / probs.sum()
        indices = torch.multinomial(probs, batch_size, replacement=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class ElasticWeightConsolidation:
    """Protect important weights when learning new tasks"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher: dict = {}
        self.old_params: dict = {}

    def compute_fisher(self, dataloader, n_samples: int = 1000):
        """Calculate which weights are most important for current tasks"""
        for name, param in self.model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
            self.old_params[name] = param.clone().detach()

        self.model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            self.model.zero_grad()
            output = self.model(batch["input_ids"])
            if isinstance(output, tuple):
                loss = output[0]
            else:
                loss = output.mean()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.detach() ** 2

            count += 1

        self.model.train()
        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

    def regularized_loss(self, base_loss: torch.Tensor, lambda_ewc: float = 100.0) -> torch.Tensor:
        """Add penalty for changing important weights"""
        ewc_loss = torch.tensor(0.0, device=base_loss.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name].to(param.device)
                old = self.old_params[name].to(param.device)
                ewc_loss += (fisher * (param - old) ** 2).sum()
        return base_loss + (lambda_ewc / 2) * ewc_loss


class OnlineDataStreamer:
    """Stream new data continuously during training"""

    def __init__(self, data_sources: list):
        self.sources = data_sources
        self.current_source = 0

    def stream_batch(self, batch_size: int):
        source = self.sources[self.current_source]
        batch = next(iter(source))
        self.current_source = (self.current_source + 1) % len(self.sources)
        return batch


class LifelongTrainer:
    """Training loop that prevents catastrophic forgetting"""

    def __init__(self, model: nn.Module, optimizer, replay_capacity: int = 200000):
        self.model = model
        self.optimizer = optimizer
        self.buffer = ExperienceReplayBuffer(capacity=replay_capacity)
        self.ewc = ElasticWeightConsolidation(model)

    def train_step(self, new_batch, lambda_ewc: float = 100.0):
        input_ids = new_batch["input_ids"]

        # Mix with replay
        if len(self.buffer) > 0:
            replay = self.buffer.sample(min(input_ids.shape[0] // 2, len(self.buffer)))
            replay_ids = torch.cat([b["input_ids"] for b in replay], dim=0)
            input_ids = torch.cat([input_ids, replay_ids], dim=0)

        output = self.model(input_ids)
        if isinstance(output, tuple):
            loss = output[0]
        else:
            loss = output.mean()

        loss = self.ewc.regularized_loss(loss, lambda_ewc=lambda_ewc)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.buffer.add(new_batch, importance=loss.item())
        return loss.item()
