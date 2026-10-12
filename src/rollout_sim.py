"""
Closed-loop evaluation in the gym-pusht simulator.
Every rule sees the same env seeds.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import numpy as np
import torch

from config import Config, DiffusionConfig
from diffusion import DiffusionPolicy
from replanning import ReplanningStrategy, FixedReplanning
from train import load_policy, set_all_seeds

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)
logging.getLogger("src").setLevel(logging.INFO)

ENV_ID = "gym_pusht/PushT-v0"
MAX_ROLLOUT_STEPS = 300  # the gym-pusht default; the reference eval uses a similar cap
FIRST_EVAL_SEED = 100_000  # disjoint from anything used for training / splitting




# ----------------------------------------------------------------
# Helpers for the simulation enviroment + model conditioning 
# ----------------------------------------------------------------

def make_simulation_env(max_steps: int = MAX_ROLLOUT_STEPS) -> gym.Env:
    # "environment_state_agent_pos" = the keypoint observation that match lerobot/pusht_keypoints
    return gym.make(ENV_ID, 
                    obs_type="environment_state_agent_pos", 
                    render_mode=None,
                    max_episode_steps=max_steps
                    )


def format_sim_observations(sim_observations: dict, cfg: Config) -> np.ndarray:
    agent_pos = np.asarray(sim_observations["agent_pos"], dtype=np.float32).reshape(-1)
    block_keypoints = np.asarray(sim_observations["environment_state"], dtype=np.float32).reshape(-1)
    if agent_pos.shape != (cfg.data.agent_dim,) or block_keypoints.shape != (cfg.data.keypoint_dim,):
        raise ValueError(f"env obs shapes {agent_pos.shape}/{block_keypoints.shape} do not match the dataset ({cfg.data.agent_dim},)/({cfg.data.keypoint_dim},)")
    return np.concatenate([agent_pos, block_keypoints])


class ObservationContext:
    def __init__(self, obs_horizon: int) -> None:
        self.obs_horizon = obs_horizon
        self._frames = collections.deque(maxlen=obs_horizon)

    def init_rollout(self, initial_obs: np.ndarray) -> None:
        """Pads the start of the rollout with copies of the first frame."""
        self._frames.clear()
        for _ in range(self.obs_horizon):
            self._frames.append(initial_obs)

    def update(self, step_obs: np.ndarray) -> None:
        """Slides the context window forward by one step."""
        self._frames.append(step_obs)

    def get_batch_tensor(self, device: torch.device) -> torch.Tensor:
        """Returns batched context tensor: (1, T_o, obs_dim)."""
        return torch.from_numpy(np.stack(self._frames)).unsqueeze(0).to(device)
        
      

# ---------------------------------------------------------------------------
# Infrastructure to rollout a SINGLE simulation given a random start seed
# ---------------------------------------------------------------------------

@dataclass
class RolloutMetrics:
    seed: int
    success: bool
    n_total_steps: int
    n_planning_cycles: int 
    n_forward_passes: int      # replans x n_samples x denoising steps
    max_coverage_score: float 
    duration_sec: float
    n_actions_per_planning_cycle: list[int] = field(default_factory=list)


@torch.no_grad()
def run_rollout_sim(
    env: gym.Env,
    policy: DiffusionPolicy,
    rule: ReplanningStrategy,
    cfg: Config,
    seed: int,
    max_steps: int = MAX_ROLLOUT_STEPS,
) -> RolloutMetrics:
    
    # New generator each rollout so the noise stream is a function of the seed only
    generator = torch.Generator().manual_seed(seed)
    device = policy.device

    env_obs, _ = env.reset(seed=seed)
    previous_obs_context = ObservationContext(cfg.data.obs_horizon)
    previous_obs_context.init_rollout(format_sim_observations(env_obs, cfg))

    n_total_steps = 0
    n_planning_cycles = 0
    max_coverage_score = 0.0
    success = False
    n_actions_per_planning_cycle = []
    t_start = time.perf_counter()
    done = False

    while not done and n_total_steps < max_steps:
        planned_actions = policy.sample(
            previous_obs_context.get_batch_tensor(device), 
            n_samples=rule.n_samples, 
            generator=generator
        )[0]

        n_actions_to_take = rule.compute_commit_length(planned_actions)
        action_sequence_taken = planned_actions[0, :n_actions_to_take].cpu().numpy()
        n_planning_cycles += 1
        n_actions_per_planning_cycle.append(n_actions_to_take)

        for action in action_sequence_taken:
            action = np.clip(action, cfg.data.box_lo, cfg.data.box_hi)
            env_obs, step_reward, terminated, truncated, _ = env.step(action)

            previous_obs_context.update(format_sim_observations(env_obs, cfg))
            n_total_steps += 1
            max_coverage_score = max(max_coverage_score, float(step_reward))

            if terminated:
                success = True
            if terminated or truncated or n_total_steps >= max_steps:
                done = True
                break

    return RolloutMetrics(
            seed=seed, 
            success=success, 
            n_total_steps=n_total_steps, 
            n_planning_cycles=n_planning_cycles,
            n_forward_passes=(n_planning_cycles * rule.n_samples * policy.num_inference_steps),
            max_coverage_score=max_coverage_score, 
            duration_sec=time.perf_counter() - t_start,
            n_actions_per_planning_cycle=n_actions_per_planning_cycle,
        )



# ------------------------------------------------------
# Infrastructure to rollout multiple simulations
# ------------------------------------------------------

@dataclass
class RolloutBatchSummary:
    rule_name: str               
    n_rollouts: int              
    success_rate: float
    success_se: float            
    mean_total_steps: float      
    mean_planning_cycles: float  
    mean_forward_passes: float
    mean_max_coverage_score: float         
    mean_duration_sec: float               
    mean_actions_per_planning_cycle: float


def run_rollout_batch(
    policy: DiffusionPolicy,
    rule: ReplanningStrategy,
    cfg: Config,
    seeds: list[int],
    rule_name: str,
    verbose: bool = True,
) -> tuple[RolloutBatchSummary, list[RolloutMetrics]]:
    logger.info("Starting batch rollouts...")
    env = make_simulation_env()
    validate_env_config_alignment(env, cfg)

    results = []
    for i, seed in enumerate(seeds):
        rollout_result = run_rollout_sim(env, policy, rule, cfg, seed)
        results.append(rollout_result)

        if verbose:
            wins = sum(x.success for x in results)
            logger.info(f"[{rule_name}] rollout {i + 1:>3}/{len(seeds)}  seed={seed}  "
                  f"{'OK  ' if rollout_result.success else 'fail'}  steps={rollout_result.n_total_steps:>3}  cycles={rollout_result.n_planning_cycles:>3}  "
                  f"max_cov={rollout_result.max_coverage_score:.2f}  running success={wins / len(results):.0%}")
    env.close()
    logger.info("Batch rollouts complete.")

    n = len(results)
    p = sum(r.success for r in results) / n
    all_executed_lengths = [x for r in results for x in r.n_actions_per_planning_cycle]

    summary = RolloutBatchSummary(
        rule_name=rule_name, 
        n_rollouts=n, 
        success_rate=p, 
        success_se=float(np.sqrt(p * (1 - p) / n)),        
        mean_total_steps=float(np.mean([r.n_total_steps for r in results])),
        mean_planning_cycles=float(np.mean([r.n_planning_cycles for r in results])),
        mean_forward_passes=float(np.mean([r.n_forward_passes for r in results])),
        mean_max_coverage_score=float(np.mean([r.max_coverage_score for r in results])),
        mean_duration_sec=float(np.mean([r.duration_sec for r in results])),
        mean_actions_per_planning_cycle=float(np.mean(all_executed_lengths)),
    )
    return summary, results



# ------------------------------------------------------
# Misc helpers + I/O + CLI
# ------------------------------------------------------

def validate_env_config_alignment(env: gym.Env, cfg: Config) -> None:
    """Ensure no train config mismatches."""
    fps = env.metadata.get("render_fps")
    if fps is not None and fps != cfg.data.fps:
        logger.warning(f"Env render_fps={fps} but the dataset/config fps={cfg.data.fps}. "
              f"Actions will be executed at a different rate than what was trained on.")
              
    env_obs, _ = env.reset(seed=0)
    obs_vector = format_sim_observations(env_obs, cfg)
    
    lo, hi = cfg.data.box_lo, cfg.data.box_hi
    if obs_vector.min() < lo - 20 or obs_vector.max() > hi + 20:
        raise ValueError(
            f"Env observation outside the dataset's coordinate range: "
            f"[{obs_vector.min():.1f}, {obs_vector.max():.1f}]"
        )


def save_batch_results(path: Path, summary: RolloutBatchSummary, rollouts: list[RolloutMetrics], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "meta": meta, 
        "summary": asdict(summary), 
        "rollouts": [asdict(r) for r in rollouts]
    }
    path.write_text(json.dumps(output_data, indent=1))
    logger.info(f"Results saved to: {path}")


def log_batch_summary(s: RolloutBatchSummary) -> None:
    logger.info(
        f"Success {s.success_rate:.1%} +/- {s.success_se:.1%} (n={s.n_rollouts})  "
        f"max_cov {s.mean_max_coverage_score:.3f}  steps {s.mean_total_steps:.0f}  "
        f"cycles {s.mean_planning_cycles:.1f}  fwd passes {s.mean_forward_passes:.0f}  "
        f"actions/cycle {s.mean_actions_per_planning_cycle:.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to override default config JSON")
    p.add_argument("--ckpt", required=True, help="Path to the model checkpoint file.")
    p.add_argument("--raw", action="store_true", help="Use raw network weights instead of EMA weights.")
    p.add_argument("--n-rollouts", type=int, default=50, help="Number of rollouts to run per batch.")
    p.add_argument("--seed0", type=int, default=FIRST_EVAL_SEED, help="Starting environment seed.")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default=None, help="Set the diffusion sampler.")
    p.add_argument("--steps", type=int, default=None, help="Override the number of inference steps.")
    p.add_argument("--device", default="auto")

    subparsers = p.add_subparsers(dest="strategy", required=True, help="Replanning strategy to use.")

    fixed_p = subparsers.add_parser("fixed", help="Run replanning using the fixed action horizon rule.")
    fixed_p.add_argument("--action-horizon", type=int, nargs='+', default=None, help="Override action horizon. Space separated for manual sweep.")
    fixed_p.add_argument("--sweep", action="store_true", help="Run a sweep over all action horizons defined in config.")

    args = p.parse_args()

    # Load policy + master config
    policy, cfg = load_policy(args.ckpt, device=args.device, use_ema=not args.raw)
    set_all_seeds(0)

    if args.config is not None:
        with open(args.config, 'r') as f:
            override_dict = json.load(f)
        cfg = Config.from_checkpoint_with_override(cfg.to_dict(), override_dict)
    
    # Global CLI overrides
    if args.sampler is not None or args.steps is not None:
        sampler = args.sampler or cfg.diffusion.sampler
        steps = args.steps or cfg.diffusion.num_inference_steps
        new_diff_cfg = dataclasses.replace(cfg.diffusion, sampler=sampler, num_inference_steps=steps)
        policy.set_sampler(new_diff_cfg)
        cfg.diffusion = new_diff_cfg   
    sampler_tag = f"{cfg.diffusion.sampler}{policy.num_inference_steps}"
    logger.info(f"Loaded {args.ckpt} ({'raw' if args.raw else 'EMA'} weights)  sampler={sampler_tag}  device={policy.device}")

    # Subparser resolution for future multiple replanning strategies
    replanning_strategies_to_run = []    
    if args.strategy == "fixed":
        strat_cfg = cfg.replanning.fixed

        if args.action_horizon is not None:
            action_horizon_values = args.action_horizon
        elif args.sweep:
            action_horizon_values = strat_cfg.action_horizon_sweep
        else:
            action_horizon_values = [strat_cfg.action_horizon]

        for action_horizon in action_horizon_values:
            replanning_strategies_to_run.append((FixedReplanning(action_horizon=action_horizon), f"fixed_action_horizon{action_horizon}"))

    # Future replanning strategy
    # elif args.strategy == "alt":
    #    replanning_strategies_to_run.append()


    # Execute rollouts
    seeds = [args.seed0 + i for i in range(args.n_rollouts)]
    out_dir = Path(args.ckpt).parent / "rollouts"
    out_dir.mkdir(exist_ok=True)
    
    for rule, rule_name in replanning_strategies_to_run:
        logger.info(f"Evaluating {rule_name}:")
        summary, rollouts = run_rollout_batch(
            policy=policy, 
            rule=rule, 
            cfg=cfg, 
            seeds=seeds, 
            rule_name=rule_name
        )
        log_batch_summary(summary)
        
        save_path = out_dir / f"{rule_name}_{sampler_tag}_n{args.n_rollouts}.json"
        metadata = {
            "ckpt": str(args.ckpt), 
            "eval_args": vars(args),   
            "config": cfg.to_dict()    
        }
        save_batch_results(save_path, summary, rollouts, meta=metadata)


if __name__ == "__main__":
    main()