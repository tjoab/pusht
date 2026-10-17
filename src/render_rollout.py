from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from rollout_sim import ENV_ID
from rollout_trace import RolloutTrace 


def transform_coordinates(xy: np.ndarray, scale: float) -> list[tuple[float, float]]:
    return [(float(x * scale), float(y * scale)) for x, y in xy]


def render_trace_to_video(
    trace_path: str | Path, 
    output_video_path: str | Path, 
    fps: int = 10, 
    max_ensemble_lines: int = 16
) -> None:
    trace = RolloutTrace.load(trace_path)
    
    env = gym.make(
        ENV_ID, 
        obs_type="environment_state_agent_pos", 
        render_mode="rgb_array",
        max_episode_steps=trace.max_steps
    )
    env.reset(seed=trace.environment_seed)

    frames = []
    worst_observation_drift = 0.0

    # Replay loop across the recorded actions in the trace
    for t, action in enumerate(trace.actions):
        frame = env.render()
        h, w = frame.shape[:2]
        scale = w / 512.0
        
        # Find which replanning cycle was active at time step t
        k = int(np.searchsorted(trace.replan_indices, t, side="right")) - 1
        k = max(0, k)
        
        cycle_start_step = int(trace.replan_indices[k])
        elapsed_in_cycle = t - cycle_start_step
        action_horizon = int(trace.action_horizons[k])

        base_image = Image.fromarray(frame).convert("RGBA")
        overlay_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)

        # Draw ensemble uncertainty dispersion if its available
        if trace.ensemble_chunks.ndim == 4 and k < len(trace.ensemble_chunks):
            for sample_trajectory in trace.ensemble_chunks[k][:max_ensemble_lines]:
                draw.line(
                    transform_coordinates(sample_trajectory[elapsed_in_cycle:], scale),
                    fill=(42, 111, 187, 70), 
                    width=2
                )

        # Draw planned vs executed action chunks
        if k < len(trace.executed_chunk):
            executed_chunk = trace.executed_chunk[k]
            # Remaining full plan
            draw.line(
                transform_coordinates(executed_chunk[elapsed_in_cycle:], scale),
                fill=(217, 115, 26, 90), 
                width=3
            )
            # Portion committed to execution before next replan
            if elapsed_in_cycle < action_horizon:
                draw.line(
                    transform_coordinates(executed_chunk[elapsed_in_cycle:action_horizon], scale),
                    fill=(217, 115, 26, 255), 
                    width=3
                )

        current_coverage = trace.goal_coverage[t - 1] if t > 0 else 0.0
        status_text = (
            f"Step {t} | Replan {k} | Executing {elapsed_in_cycle + 1}/{action_horizon} | "
            f"Coverage: {current_coverage:.2f}"
        )
        draw.text((8, 8), status_text, fill=(0, 0, 0, 255))

        composited_frame = Image.alpha_composite(base_image, overlay_layer).convert("RGB")
        frames.append(np.asarray(composited_frame))

        # Step environment forward and check simulation determinism drift
        obs_dict, *_ = env.step(action)
        replay_obs = np.concatenate([
            np.asarray(obs_dict["agent_pos"]).ravel(),
            np.asarray(obs_dict["environment_state"]).ravel()
        ])
        
        if t + 1 < len(trace.obs):
            drift = float(np.abs(replay_obs - trace.obs[t + 1]).max())
            worst_observation_drift = max(worst_observation_drift, drift)

    env.close()
    
    print(f"Replay max |obs error| vs recorded: {worst_observation_drift:.4f}")
    
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_video_path, frames, fps=fps, codec="libx264", out_pixel_format="yuv420p")
    print(f"Successfully wrote video to {output_video_path} ({len(frames)} frames | Success: {trace.success})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=str, help="Path to rollout_{env_seed}_{diff_seed}.npz file")
    parser.add_argument("--out", required=True, type=str, help="Path to output .mp4 file")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    args = parser.parse_args()

    render_trace_to_video(args.trace, args.out, fps=args.fps)


if __name__ == "__main__":
    main()