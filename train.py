import argparse
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment import DoublePendulumEnv


class MetricsCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.metrics = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = sum([ep["r"] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
            self.metrics.append({
                "timesteps": self.num_timesteps,
                "mean_reward": mean_reward
            })
        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_type", type=str, default="shaped", choices=["baseline", "shaped"])
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--save_path", type=str, default="models/ppo_model.zip")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_path = f"logs/training_metrics_{args.reward_type}.csv"

    env = DoublePendulumEnv(reward_type=args.reward_type)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    callback = MetricsCallback(log_path=log_path)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")
    print(f"Logs saved to {log_path}")
    env.close()


if __name__ == "__main__":
    main()