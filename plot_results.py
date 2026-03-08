import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    baseline_path = "logs/training_metrics_baseline.csv"
    shaped_path = "logs/training_metrics_shaped.csv"

    plt.figure(figsize=(12, 6))

    if os.path.exists(baseline_path):
        df_baseline = pd.read_csv(baseline_path)
        plt.plot(df_baseline["timesteps"], df_baseline["mean_reward"], label="Baseline Reward", color="steelblue", linewidth=2)
    else:
        print(f"Baseline log not found at {baseline_path}")

    if os.path.exists(shaped_path):
        df_shaped = pd.read_csv(shaped_path)
        plt.plot(df_shaped["timesteps"], df_shaped["mean_reward"], label="Shaped Reward", color="darkorange", linewidth=2)
    else:
        print(f"Shaped log not found at {shaped_path}")

    plt.title("PPO Learning Curves: Baseline vs Shaped Reward", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_comparison.png", dpi=150)
    plt.close()
    print("Plot saved to reward_comparison.png")


if __name__ == "__main__":
    main()