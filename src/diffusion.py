from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from config import Config, DiffusionConfig
from model import ConditionalUnet1D
from normalizer import BoxNormalizer

import copy

# ---------------------------------------------------------
# Helpers for noise sampling within the diffusion policy
# ---------------------------------------------------------

def make_diffusion_scheduler(cfg: DiffusionConfig) -> DDPMScheduler | DDIMScheduler:
    common_params = dict(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        prediction_type=cfg.prediction_type,
    )
    if cfg.sampler == "ddpm":
        return DDPMScheduler(variance_type=cfg.variance_type, **common_params)
    return DDIMScheduler(**common_params)


def get_extra_step_kwargs(cfg: DiffusionConfig) -> dict:
    return {"eta": cfg.ddim_eta} if cfg.sampler == "ddim" else {}

# Using torch generators here so that noise draws are actually replicable. 
# The resulting noise is then moved over to to where ever the model is running.
def _draw_noise(shape, *, device, dtype, generator):
    gdev = generator.device if generator is not None else device
    return torch.randn(shape, device=gdev, dtype=dtype, generator=generator).to(device)

def _draw_t(high, shape, *, device, generator):
    gdev = generator.device if generator is not None else device
    return torch.randint(0, high, shape, device=gdev, generator=generator).to(device)



# ---------------------------------------------------------
# Diffusion architecture using the conditional 1D Unet 
# ---------------------------------------------------------

class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler | DDIMScheduler,
        obs_dim: int = 18,
        action_dim: int = 2,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_inference_steps: int | None = None,
        step_kwargs: dict | None = None,
        bbox: tuple[float, float] = (0.0, 512.0),
    ) -> None:
        super().__init__()
        if noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("compute_loss assumes epsilon prediction")

        self.model = model
        self.normalizer = BoxNormalizer(*bbox)

        self.scheduler = noise_scheduler
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps or self.num_train_timesteps
        self.step_kwargs = dict(step_kwargs or {})

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

    @classmethod
    def from_config(cls, cfg: Config) -> DiffusionPolicy:
        unet = ConditionalUnet1D(
            input_dim=cfg.data.action_dim,
            cond_dim=cfg.data.cond_dim,
            diffusion_timestep_embed_dim=cfg.unet.diffusion_timestep_embed_dim,
            down_dims=list(cfg.unet.down_dims),
            kernel_size=cfg.unet.kernel_size,
            n_groups=cfg.unet.n_groups,
            cond_predict_scale=cfg.unet.cond_predict_scale,
            use_attention=cfg.unet.use_attention,
            attention_n_heads=cfg.unet.attention_n_heads,
        )
        return cls(
            model=unet,
            noise_scheduler=make_diffusion_scheduler(cfg.diffusion),
            obs_dim=cfg.data.obs_dim,
            action_dim=cfg.data.action_dim,
            obs_horizon=cfg.data.obs_horizon,
            pred_horizon=cfg.data.pred_horizon,
            num_inference_steps=cfg.diffusion.num_inference_steps,
            step_kwargs=get_extra_step_kwargs(cfg.diffusion),
            bbox=(cfg.data.box_lo, cfg.data.box_hi),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


    # Generates the conditioning vector by normalizing the observations and reshaping
    def _generate_cond(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or tuple(obs.shape[1:]) != (self.obs_horizon, self.obs_dim):
            raise ValueError(f"obs must be (B, {self.obs_horizon}, {self.obs_dim}), got {obs.shape}")
        
        obs_normed = self.normalizer.normalize(obs)
        obs_out = obs_normed.reshape(obs.shape[0], -1)
        return obs_out


    # Main helper for training --> computes loss on the Unet's predicted noise 
    # (masking the loss so only real actions contribute)
    def compute_loss(self, batch: dict[str, torch.Tensor], generator: torch.Generator | None = None) -> torch.Tensor:
        future_actions = batch["actions"]
        B = future_actions.shape[0]
        if tuple(future_actions.shape[1:]) != (self.pred_horizon, self.action_dim):
            raise ValueError(f"actions must be (B, {self.pred_horizon}, {self.action_dim}), got {future_actions.shape}")

        future_actions_0 = self.normalizer.normalize(future_actions)  
        previous_observations = self._generate_cond(batch["obs"])

        # Pick a random noise level per example --> generate the noisy version
        noise = _draw_noise(future_actions_0.shape, device=future_actions_0.device, dtype=future_actions_0.dtype, generator=generator)
        t = _draw_t(self.num_train_timesteps, (B,), device=future_actions_0.device, generator=generator)
        future_actions_t = self.scheduler.add_noise(future_actions_0, noise, t)

        predicted_noise = self.model(future_actions_t, t, previous_observations)

        loss = (predicted_noise - noise).pow(2)
        valid = (~batch["is_actions_pad"].bool()).unsqueeze(-1).to(loss.dtype)
        n_valid = (valid.sum() * loss.shape[-1]).clamp_min(1.0)
        masked_loss = (loss * valid).sum() / n_valid
        return masked_loss


    # Main helper for sampling --> generates random noise and diffuses it into the the predicted future actions
    @torch.no_grad()
    def sample(self, obs: torch.Tensor,n_samples: int = 1,generator: torch.Generator | None = None) -> torch.Tensor:
        B = obs.shape[0]
        previous_observations = self._generate_cond(obs).repeat_interleave(n_samples, dim=0)

        x = _draw_noise(
            (B * n_samples, self.pred_horizon, self.action_dim),
            device=previous_observations.device, 
            dtype=previous_observations.dtype, 
            generator=generator
        )

        self.scheduler.set_timesteps(self.num_inference_steps)
        for t in self.scheduler.timesteps:
            t_batch = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch, previous_observations)
            x = self.scheduler.step(predicted_noise, t, x, generator=generator, **self.step_kwargs).prev_sample

        predicted_future_actions = self.normalizer.unnormalize(x)
        predicted_future_actions = predicted_future_actions.reshape(B, n_samples, self.pred_horizon, self.action_dim)
        return predicted_future_actions


    # Helper for changing the sampling method used to sample (i.e. between DDPM or DDIM)
    def set_sampler(self, diffusion_config: DiffusionConfig) -> None:
        new_scheduler = make_diffusion_scheduler(diffusion_config)
        current_scheduler = self.scheduler
        
        if new_scheduler.betas.shape != current_scheduler.betas.shape or not torch.allclose(new_scheduler.betas, current_scheduler.betas):
            raise ValueError("New sampler's beta schedule differs from the one the model was trained with.")
            
        self.scheduler = new_scheduler
        self.num_inference_steps = diffusion_config.num_inference_steps or diffusion_config.num_train_timesteps
        self.step_kwargs = get_extra_step_kwargs(diffusion_config)


# ---------------------------------------------------------
# Helpers for validation of the model
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_loss(
    policy: DiffusionPolicy, 
    data_loader: DataLoader, 
    seed: int = 0, 
    max_batches: int | None = None
) -> float:
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = policy.device
    
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss += policy.compute_loss(batch, generator=generator).item()
        batch_count += 1
        
    return total_loss / max(batch_count, 1)


# ------------------------------------------------------------------------
# EMA object for use in training to help with noisy updates + sampling
# ------------------------------------------------------------------------

class EMA:
    def __init__(
        self,
        live_policy: nn.Module,
        decay_power: float = 0.75,
        decay_scale: float = 1.0,
        min_decay: float = 0.0,
        max_decay: float = 0.9999,
        update_after_step: int = 0,
    ) -> None:
        self.shadow_policy = copy.deepcopy(live_policy)
        self.shadow_policy.eval()
        self.shadow_policy.requires_grad_(False)

        self.decay_power = decay_power
        self.decay_scale = decay_scale
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.update_after_step = update_after_step
        self.optimization_step = 0 

    def get_decay(self, optimization_step: int) -> float:
        steps_since_start = max(0, optimization_step - self.update_after_step - 1)
        if steps_since_start <= 0:
            return 0.0
        
        ramped_decay = 1.0 - (1.0 + steps_since_start / self.decay_scale) ** -self.decay_power
        return max(self.min_decay, min(ramped_decay, self.max_decay))

    @torch.no_grad()
    def update(self, live_policy: nn.Module) -> None:
        decay = self.get_decay(self.optimization_step)

        shadow_params = list(self.shadow_policy.parameters())
        live_params = list(live_policy.parameters())
        if len(shadow_params) != len(live_params):
            raise RuntimeError("Shadow and live policies have different parameter counts")

        for shadow_param, live_param in zip(shadow_params, live_params):
            if decay == 0.0:
                shadow_param.copy_(live_param)
            else:
                # shadow = decay * shadow + (1 - decay) * live 
                # NOTE: this is inplace
                shadow_param.mul_(decay).add_(live_param.detach(), alpha=1.0 - decay)

        for shadow_buffer, live_buffer in zip(self.shadow_policy.buffers(), live_policy.buffers()):
            shadow_buffer.copy_(live_buffer)

        self.optimization_step += 1

    def state_dict(self) -> dict:
        return {
            "shadow_policy": self.shadow_policy.state_dict(),
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow_policy.load_state_dict(state["shadow_policy"])
        self.optimization_step = state["optimization_step"]