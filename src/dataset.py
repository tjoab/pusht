import random
import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import logging
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR) 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_episodes(
        val_frac: float = 0.1, 
        seed: int = 0
    ) -> tuple[list[int], list[int]]:

    """Episode level splitting for train and validation."""
    data = LeRobotDataset("lerobot/pusht_keypoints") 

    n_episodes = data.num_episodes
    episode_ids = list(range(n_episodes))
    
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    n_val = int(n_episodes * val_frac)

    train_ids, val_ids = episode_ids[n_val:], episode_ids[:n_val]
    return train_ids, val_ids



class PushTKeypointsDataset(Dataset):
    def __init__(
            self, 
            episodes: list[int], 
            obs_horizon: int = 2, 
            pred_horizon: int = 16, fps: int = 10, 
            preload: bool = False
        ) -> None:
        """
        Training data for PushT task, where observations comes from fixed keypoints on T.
        """
        dt = 1.0 / fps

        # GOAL: the modeling plan is --> given recent observations, learn to generate a 
        # plausible sequence of FUTURE actions —-> i.e an entire short horizon plan.

        # Training example needs to look like:
        # (short window of past observations, the actual action sequence a human performed right after)

        # `delta_timestamps`` is how we tell LeRobotDataset to assemble that window for us, purely 
        # by specifying TIME OFFSETS relative to whatever frame index we ask for, i.e.
        #       "observation.state"             : [-0.1, 0.0]            (current and previous To-1)
        #       "observation.environment_state" : [-0.1, 0.0]            (current and previous To-1)
        #       "action"                        : [0.0, 0.1, ..., 1.5]   (current and next Tp-1)
        delta_timestamps = {
            "observation.state": [-i * dt for i in reversed(range(obs_horizon))],
            "observation.environment_state": [-i * dt for i in reversed(range(obs_horizon))],
            "action": [i * dt for i in range(pred_horizon)],
        }
        self.data = LeRobotDataset(
            "lerobot/pusht_keypoints",
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
        self._preloaded_data = self._preload_dataset() if preload else None


    def _load_item(self, idx: int) -> dict[str, torch.tensor]:
        item = self.data[idx]

        # Joining the state of the robot (i.e. current location) with the state of the env (i.e. the 
        # coords of the fixed keypoints on the T) as a single past context
        previous_observations_torch = torch.cat(
            [item["observation.state"], item["observation.environment_state"]],
            dim=-1
        )
        # LeRobotDataset already returns this as a tensor
        future_actions_torch = item["action"]

        # There is another thing to handle here --> edge cases... literally
        # We cant fetch timesteps before the start of an episode nor past the end. LeRobotDataset will 
        # handle this for us by also exposing mask tensors for use to use so that we can zero the loss 
        is_previous_observations_pad = item["observation.state_is_pad"] | item["observation.environment_state_is_pad"]
        is_future_actions_pad = item["action_is_pad"]

        return {
            "obs": previous_observations_torch, 
            "is_obs_pad": is_previous_observations_pad,
            "actions": future_actions_torch, 
            "is_actions_pad": is_future_actions_pad
        }


    def _preload_dataset(self) -> dict[str, torch.Tensor]:
        """Preloads all dataset items into memory and stacks them into tensors."""
        logger.info('Preloading data...')
        
        accumulated_elems = {}
        for idx in range(len(self.data)):
            for key, tensor in self._load_item(idx).items():
                accumulated_elems.setdefault(key, []).append(tensor)

        logger.info('Preloading complete.')
        return {key: torch.stack(tensors) for key, tensors in accumulated_elems.items()}


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> dict[str, torch.tensor]:
        if self._preloaded_data is not None:
            return {key: tensor[idx] for key, tensor in self._preloaded_data.items()}
        return self._load_item(idx)