from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class RolloutTraceRecorder:
    """Telemetry recorder that captures enviroment and policy details during rollout."""
    environment_seed: int
    diffusion_seed: int
    strategy_name: str
    n_samples: int
    num_inference_steps: int
    max_steps: int
    
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    goal_coverage: list[float] = field(default_factory=list)
    replan_indices: list[int] = field(default_factory=list)
    action_horizons: list[int] = field(default_factory=list)
    executed_chunk: list[np.ndarray] = field(default_factory=list)
    ensemble_chunks: list[np.ndarray] = field(default_factory=list)
    success: bool = False


    def record_env_step(self, observation: np.ndarray, action: np.ndarray, coverage: float) -> None:
        self.obs.append(observation.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.goal_coverage.append(float(coverage))


    def record_replan_event(self, step: int, action_horizon: int, samples: np.ndarray) -> None:
        self.replan_indices.append(step)
        self.action_horizons.append(action_horizon)
        self.executed_chunk.append(samples[0].astype(np.float32))
        
        if samples.shape[0] > 1:
            self.ensemble_chunks.append(samples.astype(np.float32))


    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        action_dim = self.executed_chunk[0].shape[-1] if self.executed_chunk else 2
        
        np.savez_compressed(
            path,
            environment_seed=self.environment_seed,
            diffusion_seed=self.diffusion_seed,
            strategy_name=np.array(self.strategy_name),
            n_samples=self.n_samples,
            num_inference_steps=self.num_inference_steps,
            max_steps=self.max_steps,
            success=self.success,
            obs=np.stack(self.obs),
            actions=np.stack(self.actions) if self.actions else np.zeros((0, action_dim), np.float32),
            goal_coverage=np.asarray(self.goal_coverage, dtype=np.float32),
            replan_indices=np.asarray(self.replan_indices, dtype=np.int32),
            action_horizons=np.asarray(self.action_horizons, dtype=np.int32),
            executed_chunk=np.stack(self.executed_chunk) if self.executed_chunk else np.zeros((0,), np.float32),
            ensemble_chunks=np.stack(self.ensemble_chunks) if self.ensemble_chunks else np.zeros((0,), np.float32),
        )


@dataclass
class RolloutTrace:
    """Trace object to load for offline replay, eval and comparisons."""
    environment_seed: int
    diffusion_seed: int
    strategy_name: str
    n_samples: int
    num_inference_steps: int
    max_steps: int
    success: bool
    obs: np.ndarray
    actions: np.ndarray
    goal_coverage: np.ndarray
    replan_indices: np.ndarray
    action_horizons: np.ndarray
    executed_chunk: np.ndarray
    ensemble_chunks: np.ndarray  # empty if n_samples == 1

    @property
    def steps_taken(self) -> int:
        return len(self.actions)

    @property
    def time_to_success(self) -> int:
        return self.steps_taken if self.success else self.max_steps

    def max_goal_coverage_up_to(self, cutoff_step: int) -> float:
        c = self.goal_coverage[:cutoff_step]
        return float(c.max()) if len(c) else 0.0

    def count_replan_cycles_up_to(self, cutoff_step: int) -> int:
        return int((self.replan_indices < cutoff_step).sum())

    def calculate_cumulative_forward_passes_up_to(self, cutoff_step: int) -> int:
        return self.count_replan_cycles_up_to(cutoff_step) * self.n_samples * self.num_inference_steps

    @classmethod
    def load(cls, path: str | Path) -> "RolloutTrace":
        data = np.load(path, allow_pickle=False)
        return cls(
            environment_seed=int(data["environment_seed"]),
            diffusion_seed=int(data["diffusion_seed"]),
            strategy_name=str(data["strategy_name"]),
            n_samples=int(data["n_samples"]),
            num_inference_steps=int(data["num_inference_steps"]),
            max_steps=int(data["max_steps"]),
            success=bool(data["success"]),
            obs=data["obs"],
            actions=data["actions"],
            goal_coverage=data["goal_coverage"],
            replan_indices=data["replan_indices"],
            action_horizons=data["action_horizons"],
            executed_chunk=data["executed_chunk"],
            ensemble_chunks=data["ensemble_chunks"],
        )


def get_rollout_trace_path(run_dir: Path, environment_seed: int, diffusion_seed: int) -> Path:
    return run_dir / f"rollout_{environment_seed}_{diffusion_seed}.npz"


def load_single_rollout_trace(run_dir: str | Path, environment_seed: int, diffusion_seed: int) -> RolloutTrace:
    path = get_rollout_trace_path(Path(run_dir), environment_seed, diffusion_seed)
    if not path.exists():
        raise FileNotFoundError(f"No rollout trace found at {path}")
    return RolloutTrace.load(path)


def load_rollout_traces(run_dir: str | Path) -> list[RolloutTrace]:
    run_dir = Path(run_dir)
    trace_files = sorted(run_dir.glob("rollout_*.npz"))
    if not trace_files:
        raise FileNotFoundError(f"No rollout traces found in {run_dir}")
    return [RolloutTrace.load(f) for f in trace_files]

