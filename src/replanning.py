"""
Houses replanning strategies which determine the action horizon during a rollout.
I.e. how many action steps from a predicted trajectory to actually execute on before
querying the policy to replan.
"""
from dataclasses import dataclass
from typing import Protocol
import torch


class ReplanningStrategy(Protocol):
    n_samples: int

    def compute_commit_length(self, samples: torch.Tensor) -> int: ...


@dataclass
class FixedReplanning:
    """Always execute the first `action_horizon` actions of a planned action sequence."""
    action_horizon: int
    n_samples: int = 1

    def compute_commit_length(self, samples: torch.Tensor) -> int:
        return min(self.action_horizon, samples.shape[1])