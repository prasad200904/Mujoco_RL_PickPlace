"""PPO Reinforcement Learning Training Pipeline for Pick-and-Place.

Trains a PPO agent to autonomously detect, reach, grasp, lift, and place the cube
onto the target. Logs step-by-step reward calculations and exports training metrics.

Usage:
    python main.py train --timesteps 50000
    or on macOS:
    python training/train.py --timesteps 50000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.pick_place_env import PickPlaceEnv
from environment import scene_loader as sl


class TrainingProgressCallback(BaseCallback):
    """Logs detailed step-by-step reward calculations and training progress."""

    def __init__(self, check_freq: int = 1000, log_dir: Path | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir or (PROJECT_ROOT / "results" / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.success_history: list[bool] = []
        self.timesteps_history: list[int] = []
        self.moving_success_rates: list[float] = []

        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        self.total_episodes = 0
        self.last_print_time = time.time()

    def _on_step(self) -> bool:
        # Extract reward and info from current step
        reward = float(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        self.current_ep_reward += reward
        self.current_ep_length += 1

        if done:
            self.total_episodes += 1
            is_success = bool(info.get("is_success", False))

            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.success_history.append(is_success)
            self.timesteps_history.append(self.num_timesteps)

            # Compute moving average success rate over last 20 episodes
            window = self.success_history[-20:]
            success_rate = (sum(window) / len(window)) * 100.0
            self.moving_success_rates.append(success_rate)

            # Periodic console feedback of reward calculations
            now = time.time()
            if now - self.last_print_time >= 1.0 or self.total_episodes % 10 == 0:
                dist_reach = info.get("dist_reach", 0.0)
                dist_place = info.get("dist_place", 0.0)
                print(
                    f"[Ep {self.total_episodes:4d} | Step {self.num_timesteps:6d}] "
                    f"Ep Reward: {self.current_ep_reward:7.2f} | "
                    f"Success Rate: {success_rate:5.1f}% | "
                    f"Reach: {dist_reach*100:4.1f}cm | "
                    f"Target Dist: {dist_place*100:4.1f}cm | "
                    f"Ep Len: {self.current_ep_length}"
                )
                self.last_print_time = now

            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        return True

    def save_history(self) -> Path:
        """Save training history to JSON for graph plotting."""
        history_path = self.log_dir / "training_history.json"
        data = {
            "timesteps": self.timesteps_history,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "moving_success_rates": self.moving_success_rates,
            "total_episodes": self.total_episodes,
        }
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Training metrics saved to: {history_path}")
        return history_path


def train(
    timesteps: int = 50000,
    seed: int = 42,
    config_path: str | None = None,
) -> Path:
    """Run PPO training and save model and metrics."""
    cfg = sl.load_config(config_path)
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    logs_dir = PROJECT_ROOT / cfg["paths"]["logs_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  [STARTING REINFORCEMENT LEARNING TRAINING (PPO)]")
    print(f"  Total Timesteps: {timesteps}")
    print(f"  Random Seed:     {seed}")
    print(f"  Environment:     PickPlaceEnv (24-dim Obs, 4-dim Cartesian Action)")
    print("=" * 70 + "\n")

    # Instantiate environment
    raw_env = PickPlaceEnv(config=cfg)
    env = Monitor(raw_env, str(logs_dir / "monitor.csv"))

    # PPO Hyperparameters tuned for continuous robotic manipulation
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    callback = TrainingProgressCallback(log_dir=logs_dir)

    try:
        model.learn(total_timesteps=timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")

    # Save trained policy
    model_path = models_dir / "ppo_pick_place.zip"
    model.save(str(model_path))
    print(f"\n[DONE] Trained model saved to: {model_path}")

    # Save metrics JSON
    callback.save_history()

    env.close()
    return model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO agent on Pick-and-Place")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total environment steps to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(timesteps=args.timesteps, seed=args.seed, config_path=args.config)

