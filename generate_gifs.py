import os
import argparse
import numpy as np
import imageio
import pygame
from stable_baselines3 import PPO
from environment import DoublePendulumEnv


def capture_frame(screen):
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    return frame


def record_agent(model_path, gif_path, max_steps=300):
    env = DoublePendulumEnv(render_mode="human")
    model = PPO.load(model_path, env=env)
    frames = []

    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()

        if env.screen is not None:
            frame = capture_frame(env.screen)
            frames.append(frame)

        done = terminated or truncated
        step += 1

    env.close()

    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"GIF saved to {gif_path}")
    else:
        print(f"No frames captured for {gif_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_model", type=str, default="models/ppo_initial.zip")
    parser.add_argument("--final_model", type=str, default="models/ppo_model.zip")
    args = parser.parse_args()

    os.makedirs("media", exist_ok=True)

    if os.path.exists(args.initial_model):
        record_agent(args.initial_model, "media/agent_initial.gif", max_steps=200)
    else:
        print(f"Initial model not found at {args.initial_model}")

    if os.path.exists(args.final_model):
        record_agent(args.final_model, "media/agent_final.gif", max_steps=500)
    else:
        print(f"Final model not found at {args.final_model}")


if __name__ == "__main__":
    main()