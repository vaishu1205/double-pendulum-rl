import argparse
import os
import numpy as np
import imageio
import pygame
from stable_baselines3 import PPO
from environment import DoublePendulumEnv


def capture_frame(screen, width, height):
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_model.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--save_gif", type=str, default=None)
    args = parser.parse_args()

    env = DoublePendulumEnv(render_mode="human")
    model = PPO.load(args.model_path, env=env)

    os.makedirs("media", exist_ok=True)

    frames = []
    record = args.save_gif is not None

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()

            if record and env.screen is not None:
                frame = capture_frame(env.screen, env.screen_width, env.screen_height)
                frames.append(frame)

            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    if record and frames:
        imageio.mimsave(args.save_gif, frames, fps=30)
        print(f"GIF saved to {args.save_gif}")

    env.close()


if __name__ == "__main__":
    main()