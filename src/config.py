from __future__ import annotations
from dataclasses import dataclass, field, asdict
import copy


@dataclass
class DataConfig:
    fps: int = 10  
    obs_horizon: int = 2  
    pred_horizon: int = 16 

    agent_dim: int = 2
    keypoint_dim: int = 16
    action_dim: int = 2

    val_frac: float = 0.1
    split_seed: int = 0
    preload: bool = False

    box_lo: float = 0.0
    box_hi: float = 512.0

    @property
    def obs_dim(self) -> int:
        return self.agent_dim + self.keypoint_dim

    @property
    def cond_dim(self) -> int:
        return self.obs_horizon * self.obs_dim

    def __post_init__(self) -> None:
        if self.obs_horizon < 1 or self.pred_horizon < 1:
            raise ValueError("obs_horizon and pred_horizon must be >= 1")
        if not (0.0 < self.val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {self.val_frac}")
        if self.box_hi <= self.box_lo:
            raise ValueError("box_hi must be > box_lo")


@dataclass
class UNetConfig:
    down_dims: tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 3
    n_groups: int = 8
    diffusion_timestep_embed_dim: int = 256
    cond_predict_scale: bool = True
    use_attention: bool = False
    attention_n_heads: int = 4

    @property
    def n_downsamples(self) -> int:
        return len(self.down_dims) - 1

    def __post_init__(self) -> None:
        self.down_dims = tuple(self.down_dims)
        if len(self.down_dims) < 2:
            raise ValueError("need at least 2 entries in down_dims")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")
        for d in self.down_dims:
            if d % self.n_groups:
                raise ValueError(f"every down_dim must be divisible by n_groups={self.n_groups}; got {d}")
        if self.down_dims[0] % 8:
            raise ValueError(f"down_dims[0] must be divisible by 8 (final conv GroupNorm); got {self.down_dims[0]}")
        if self.use_attention and self.down_dims[-1] % self.attention_n_heads:
            raise ValueError("bottleneck width must be divisible by attention_n_heads")


@dataclass
class DiffusionConfig:
    sampler: str = "ddpm"                  # can be "ddpm" or "ddim" --> only effects sampling
    num_train_timesteps: int = 100
    num_inference_steps: int | None = None  # None -> num_train_timesteps

    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    variance_type: str = "fixed_small"     # for DDPM only
    ddim_eta: float = 0.0                  # for DDIM only --> 0 = deterministic given the starting noise

    def __post_init__(self) -> None:
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError(f"sampler must be 'ddpm' or 'ddim', got {self.sampler!r}")
        if self.prediction_type != "epsilon":
            raise ValueError("only prediction_type='epsilon' is implemented in compute_loss")
        if self.num_inference_steps is not None and not (1 <= self.num_inference_steps <= self.num_train_timesteps):
            raise ValueError("num_inference_steps must be in [1, num_train_timesteps]")


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-4
    betas: tuple[float, float] = (0.95, 0.999)
    weight_decay: float = 1e-6
    grad_clip: float | None = None
    num_steps: int = 50_000
    lr_warmup_steps: int = 500
    ema_power: float = 0.75
    ema_max_decay: float = 0.9999
    seed: int = 0
    device: str = "auto"
 
    # data loading
    cache_dataset: bool = True
    num_workers: int = 0
 
    # bookkeeping (in optimizer steps)
    out_dir: str = "runs/baseline"
    log_every: int = 50
    val_every: int = 1000
    val_max_batches: int | None = None
    ckpt_every: int = 5000


@dataclass
class FixedReplanningConfig:
    action_horizon: int = 8
    action_horizon_sweep: tuple[int, ...] = (2, 4, 6, 8, 12, 16)


@dataclass
class ReplanningConfig:
    default_strategy: str = "fixed" 
    fixed: FixedReplanningConfig = field(default_factory=FixedReplanningConfig)

    @classmethod
    def from_dict(cls, plain_dict: dict) -> ReplanningConfig:
        return cls(
            default_strategy=plain_dict.get("default_strategy", "fixed"),
            fixed=FixedReplanningConfig(**plain_dict.get("fixed", {}))
        )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    replanning: ReplanningConfig = field(default_factory=ReplanningConfig)
 
    def __post_init__(self) -> None:
        pred_horizon = self.data.pred_horizon
        factor = 2 ** self.unet.n_downsamples
        if pred_horizon % factor:
            raise ValueError(
                f"pred_horizon={pred_horizon} must be divisible by 2**n_downsamples={factor} "
                f"(down_dims={self.unet.down_dims}) or the UNet skip connections will not line up."
            )
 
    def to_dict(self) -> dict:
        """Native python dict format in order to save to checkpoints."""
        return asdict(self)
 
    @classmethod
    def from_dict(cls, plain_dict: dict) -> Config:
        return cls(
            data=DataConfig(**plain_dict.get("data", {})),
            unet=UNetConfig(**plain_dict.get("unet", {})),
            diffusion=DiffusionConfig(**plain_dict.get("diffusion", {})),
            train=TrainConfig(**plain_dict.get("train", {})),
            replanning=ReplanningConfig.from_dict(plain_dict.get("replanning", {}))
        )

    @classmethod
    def from_checkpoint_with_override(cls, base_dict: dict, override_dict: dict) -> Config:
        def _deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _deep_update(d[k], v)
                else:
                    d[k] = v
    
        merged_dict = copy.deepcopy(base_dict)
        _deep_update(merged_dict, override_dict)
        return cls.from_dict(merged_dict)