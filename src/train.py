from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import time
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import PushTKeypointsDataset, split_episodes
from diffusion import DiffusionPolicy, evaluate_loss, EMA

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


# ------------------------------------------------------
# General helpers
# ------------------------------------------------------
 
def resolve_device(device_name: str | None = "auto") -> torch.device:
    allowed_devices = {"auto", "cpu", "cuda", "mps"}

    if device_name is None:
        device_name = "auto"
        
    if device_name not in allowed_devices:
        raise ValueError(f"Invalid device '{device_name}'. Must be one of: {', '.join(sorted(allowed_devices))}")

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------
# Data falvoured helpers
# ------------------------------------------------------

def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    logger.info('Creating dataloaders.')
    train_ids, val_ids = split_episodes(cfg.data.val_frac, cfg.data.split_seed)

    common_ds_params = dict(obs_horizon=cfg.data.obs_horizon, 
                            pred_horizon=cfg.data.pred_horizon,
                            fps=cfg.data.fps,
                            preload=cfg.data.preload)
    train_ds = PushTKeypointsDataset(episodes=train_ids, **common_ds_params)
    val_ds = PushTKeypointsDataset(episodes=val_ids, **common_ds_params)

    logger.info(f"       Episodes: {len(train_ids)} train / {len(val_ids)} val | Windows: {len(train_ds)} train / {len(val_ds)} val")

    if cfg.data.preload:
        num_workers = 0
        if cfg.train.num_workers > 0:
            logger.info(f"       Dataset is preloaded in memory. Overriding num_workers from {cfg.train.num_workers} to 0.")
    else:
        num_workers = cfg.train.num_workers


    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _cycle_loader(loader: DataLoader):
    while True:
        yield from loader


# ------------------------------------------------------
# Logging and I/O falvoured helpers
# ------------------------------------------------------

class CsvLog:
    def __init__(self, path: str | Path, fields: list[str], append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        needs_header = (not append) or (not self.path.exists())
        mode = "a" if append else "w"
        
        self._file = open(self.path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)

        if needs_header:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _atomic_save(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_policy(path: str | Path, device: str = 'auto', use_ema: bool = True) -> tuple[DiffusionPolicy, Config]:
    """
    Rebuilds the diffusion policy from any checkpoint placing it on `device` in eval mode.
    Loads the EMA shadow weights into the model when `use_ema=True`.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(checkpoint["config"])

    key = "ema_model" if use_ema else "policy"
    if key not in checkpoint:
        raise KeyError(f"{path} has no '{key}' weights")
    
    policy = DiffusionPolicy.from_config(cfg)
    policy.load_state_dict(checkpoint[key])
    policy.to(resolve_device(device)).eval()
    return policy, cfg


# ------------------------------------------------------
# Trainer class object
# ------------------------------------------------------

class Trainer:

    # Initialization 
    def __init__(self, cfg: Config, out_dir: str | Path, resume: str | Path | None = None) -> None:
        self.cfg = cfg
        train_cfg = cfg.train
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        set_all_seeds(train_cfg.seed)
        self.device = resolve_device(train_cfg.device)
        logger.info(f"Training will use device: {self.device}")

        self.train_loader, self.val_loader = make_loaders(cfg)
        self.steps_per_epoch = len(self.train_loader)
        
        self.policy = DiffusionPolicy.from_config(cfg).to(self.device)
        self.ema = EMA(self.policy, decay_power=train_cfg.ema_power, max_decay=train_cfg.ema_max_decay)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=train_cfg.lr, betas=tuple(train_cfg.betas), weight_decay=train_cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._compute_learning_rate_factor)

        self.step = 0
        self.best_val_loss = float("inf")
        if resume is not None:
            self._restore_training_state(resume)

        n_params = sum(p.numel() for p in self.policy.parameters())
        if resume is not None:
            remaining_steps = max(0, train_cfg.num_steps - self.step)
            remaining_epochs = remaining_steps / self.steps_per_epoch
            logger.info(
                f"Parameters: {n_params / 1e6:.2f}M | Steps/epoch: {self.steps_per_epoch} | "
                f"Resumed at step {self.step}/{train_cfg.num_steps} "
                f"(~{remaining_epochs:.0f} epochs remaining)"
            )
        else:
            total_epochs = train_cfg.num_steps / self.steps_per_epoch
            logger.info(
                f"Parameters: {n_params / 1e6:.2f}M | Steps/epoch: {self.steps_per_epoch} | "
                f"Total: {train_cfg.num_steps} steps (~{total_epochs:.0f} epochs)"
            )

        config_save_path = self.out_dir / "config.json"
        config_save_path.write_text(json.dumps(cfg.to_dict(), indent=2))
        logger.info(f"Copy of config saved to {config_save_path}.")


    # Lambda LR func for scheduling
    def _compute_learning_rate_factor(self, step: int) -> float:
        """Linear warmup --> then a cosine decay to 0."""
        train_cfg = self.cfg.train
        if step < train_cfg.lr_warmup_steps:
            return (step + 1) / max(1, train_cfg.lr_warmup_steps)
            
        progress = (step - train_cfg.lr_warmup_steps) / max(1, train_cfg.num_steps - train_cfg.lr_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


    # Checkpointing + saving
    def _export_weights_only(self) -> dict:
        return {
            "step": self.step, 
            "config": self.cfg.to_dict(), 
            "ema_model": self.ema.shadow_policy.state_dict()
        }

    def _export_resumable_state(self) -> dict:
        return {
            **self._export_weights_only(),
            "policy": self.policy.state_dict(),
            "ema_optimization_step": self.ema.optimization_step,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "best_val_ema": self.best_val_loss,
        }

    def _restore_training_state(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        current_config_dict = self.cfg.to_dict()
        
        for section in ("data", "unet", "diffusion"):
            if checkpoint["config"][section] != current_config_dict[section]:
                raise ValueError(
                    f"resume: '{section}' config differs from the checkpoint's -- "
                    f"checkpoint={checkpoint['config'][section]} current={current_config_dict[section]}"
                )
        self.policy.load_state_dict(checkpoint["policy"])
        self.ema.shadow_policy.load_state_dict(checkpoint["ema_model"])
        self.ema.optimization_step = checkpoint["ema_optimization_step"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_ema"]
        logger.info(f"Restored training state from {path}. Training will resume at step {self.step}.")


    def _save_checkpoints(self, snapshot: bool) -> None:
        _atomic_save(self._export_resumable_state(), self.out_dir / "last.pt")
        if snapshot:
            _atomic_save(self._export_weights_only(), self.out_dir / f"snap_{self.step:07d}.pt")


    # Validation runner
    def _validate(self) -> tuple[float, float]:
        """Fixed noise + timesteps, remains comparable across calls."""
        train_cfg = self.cfg.train

        self.policy.eval()
        val_loss_raw = evaluate_loss(self.policy, self.val_loader, seed=train_cfg.seed, max_batches=train_cfg.val_max_batches)
        val_loss_ema = evaluate_loss(self.ema.shadow_policy, self.val_loader, seed=train_cfg.seed, max_batches=train_cfg.val_max_batches)
        self.policy.train()

        return val_loss_raw, val_loss_ema


    # Main training loop
    def fit(self) -> None:
        train_cfg = self.cfg.train
        if self.step >= train_cfg.num_steps:
            logger.warning(f"Training already at step {self.step} >= num_steps={train_cfg.num_steps}. Exiting.")
            return
        
        train_log = CsvLog(self.out_dir / "train_log.csv", ["step", "loss", "lr", "grad_norm", "steps_per_sec"], append=(self.step > 0))
        val_log = CsvLog(self.out_dir / "val_log.csv", ["step", "val_loss_raw", "val_loss_ema"], append=(self.step > 0))

        train_iter = _cycle_loader(self.train_loader)
        self.policy.train()

        running_loss, running_grad_norm = 0.0, 0.0
        running_steps = 0
        t_log = time.perf_counter()

        try:
            logger.info("Starting training...")
            while self.step < train_cfg.num_steps:
                batch = {k: v.to(self.device) for k, v in next(train_iter).items()}

                loss = self.policy.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), train_cfg.grad_clip if train_cfg.grad_clip else float("inf")
                )
                loss_val, grad_norm_val = loss.item(), float(grad_norm)
                if not (math.isfinite(loss_val) and math.isfinite(grad_norm_val)):
                    raise RuntimeError(f"Infinite loss/grad at step {self.step + 1}: loss={loss_val} grad_norm={grad_norm_val}")

                self.optimizer.step()
                self.scheduler.step()
                self.ema.update(self.policy)
                self.step += 1

                running_loss += loss_val
                running_grad_norm += grad_norm_val
                running_steps += 1

                if self.step % train_cfg.log_every == 0:
                    now = time.perf_counter()
                    steps_per_sec = running_steps / (now - t_log)
                    t_log = now

                    row = {
                        "step": self.step,
                        "loss": running_loss / running_steps,
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": running_grad_norm / running_steps,
                        "steps_per_sec": steps_per_sec,
                    }
                    train_log.write(row)

                    eta = datetime.timedelta(seconds=int((train_cfg.num_steps - self.step) / max(steps_per_sec, 1e-9)))
                    logger.info(
                        f"       Step {self.step:>7}/{train_cfg.num_steps}: Loss {row['loss']:.4f} | LR {row['lr']:.2e} | "
                        f"|g| {row['grad_norm']:.3f} | {steps_per_sec:.1f} it/s | ETA {eta}"
                    )
                    running_loss = running_grad_norm = 0.0
                    running_steps = 0

                is_last_step = self.step == train_cfg.num_steps

                if self.step % train_cfg.val_every == 0 or is_last_step:
                    val_loss_raw, val_loss_ema = self._validate()
                    val_log.write({"step": self.step, "val_loss_raw": val_loss_raw, "val_loss_ema": val_loss_ema})
                    logger.info(f"       Val loss: Raw {val_loss_raw:.4f} | EMA {val_loss_ema:.4f}")
                    
                    if val_loss_ema < self.best_val_loss:
                        self.best_val_loss = val_loss_ema
                        _atomic_save(self._export_weights_only(), self.out_dir / "best_val_ema.pt")
                        logger.info(f"       Saving model weights associated with best val loss {self.best_val_loss:.4f} at {self.out_dir / "best_val_ema.pt"}")
                        
                    t_log = time.perf_counter()  

                if self.step % train_cfg.ckpt_every == 0 or is_last_step:
                    self._save_checkpoints(snapshot=True)
                    logger.info(f"       Saving current weights and a resumable state from step {self.step}.")

        except KeyboardInterrupt:
            self._save_checkpoints(snapshot=False)
            logger.warning("Training interupted. Saving current state for resumption.")
        finally:
            train_log.close()
            val_log.close()

        logger.info(f'Training complete at step {self.step}. Outputs can be foundin {self.out_dir}')



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to override default config JSON")
    p.add_argument("--out", type=str, default=None, help="Base output directory")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    p.add_argument("--num-steps", type=int, default=None, help="Override number of training steps")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = p.parse_args()

    # Load config/override defaults
    if args.config is not None:
        with open(args.config, 'r') as f:
            override_dict = json.load(f)
        cfg = Config.from_dict(override_dict)
    else:
        cfg = Config()

    # Some manual CLI overrides
    if args.num_steps is not None:
        cfg.train.num_steps = args.num_steps
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    # Auto resume
    if args.resume is not None:
        cfg.train.out_dir = str(Path(args.resume).parent)
    else:
        base_out = args.out if args.out is not None else cfg.train.out_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.train.out_dir = f"{base_out}_{timestamp}"

    # Train
    Trainer(cfg, cfg.train.out_dir, resume=args.resume).fit()


if __name__ == "__main__":
    main()