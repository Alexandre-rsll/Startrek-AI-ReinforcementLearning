##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-15
## File description:
## multi_seed

import argparse
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from baseline import run
from eval import evaluate

CI_Z = 1.96


def run_multi_seed(config_path, n_seeds=5):
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    policy_name = base_cfg["policy"]
    model_prefix = os.path.splitext(base_cfg.get("model_path", f"runs/{policy_name}"))[0]
    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

    all_train_rewards = []
    all_eval_rewards = []

    for seed in range(n_seeds):
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        cfg["model_path"] = f"{model_prefix}_seed{seed}.pth"

        print(f"\n{'='*55}")
        print(f"  Seed {seed} / {n_seeds - 1}")
        print(f"{'='*55}")

        train_rewards = run(cfg)
        eval_rewards = evaluate(cfg["model_path"], episodes=100, seed=seed, render=False)

        all_train_rewards.append(train_rewards)
        all_eval_rewards.append(eval_rewards)

    all_train_rewards = np.array(all_train_rewards)
    all_eval_rewards = np.array(all_eval_rewards)

    _save_csv(model_prefix, all_eval_rewards, n_seeds)
    _save_plot(model_prefix, policy_name, all_train_rewards, all_eval_rewards, n_seeds)
    _print_summary(all_eval_rewards, n_seeds, policy_name)


def _save_csv(model_prefix, all_eval_rewards, n_seeds):
    csv_path = f"{model_prefix}_multiseed.csv"
    seed_means = all_eval_rewards.mean(axis=1)
    overall_mean = seed_means.mean()
    overall_ci = CI_Z * seed_means.std() / np.sqrt(n_seeds)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mean_reward", "std_reward", "min_reward", "max_reward"])
        for seed in range(n_seeds):
            r = all_eval_rewards[seed]
            w.writerow([seed, f"{r.mean():.2f}", f"{r.std():.2f}",
                        f"{r.min():.2f}", f"{r.max():.2f}"])
        w.writerow(["overall", f"{overall_mean:.2f}", f"{seed_means.std():.2f}",
                    f"{seed_means.min():.2f}", f"{seed_means.max():.2f}"])
        w.writerow([])
        w.writerow([f"95% CI: {overall_mean:.2f} ± {overall_ci:.2f}"])

    print(f"\nCSV saved to {csv_path}")


def _save_plot(model_prefix, policy_name, all_train_rewards, all_eval_rewards, n_seeds):
    matplotlib.use("Agg")
    episodes = np.arange(1, all_train_rewards.shape[1] + 1)
    train_mean = all_train_rewards.mean(axis=0)
    train_ci = CI_Z * all_train_rewards.std(axis=0) / np.sqrt(n_seeds)
    seed_means = all_eval_rewards.mean(axis=1)
    eval_mean = seed_means.mean()
    eval_ci = CI_Z * seed_means.std() / np.sqrt(n_seeds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for rewards in all_train_rewards:
        ax.plot(episodes, rewards, alpha=0.15, linewidth=0.7, color="steelblue")
    ax.plot(episodes, train_mean, color="steelblue", linewidth=2, label="Mean")
    ax.fill_between(episodes, train_mean - train_ci, train_mean + train_ci,
                    alpha=0.3, color="steelblue", label="95% CI")
    ax.axhline(200, color="green", linestyle="--", linewidth=1, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{policy_name} — Training ({n_seeds} seeds)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.boxplot(all_eval_rewards.T, labels=[f"seed {i}" for i in range(n_seeds)],
               patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.5))
    ax.axhline(200, color="green", linestyle="--", linewidth=1, label="Solved (200)")
    ax.axhline(eval_mean, color="navy", linewidth=1.5,
               label=f"Mean: {eval_mean:.1f} ± {eval_ci:.1f} (95% CI)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Reward (100 episodes)")
    ax.set_title(f"{policy_name} — Eval distribution")
    ax.legend(fontsize=8)

    fig.suptitle(f"{policy_name} — multi-seed results (seeds 0–{n_seeds - 1})")
    fig.tight_layout()
    plot_path = f"{model_prefix}_multiseed.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def _print_summary(all_eval_rewards, n_seeds, policy_name):
    seed_means = all_eval_rewards.mean(axis=1)
    overall_mean = seed_means.mean()
    overall_ci = CI_Z * seed_means.std() / np.sqrt(n_seeds)

    print(f"\n{'='*55}")
    print(f"  {policy_name} — multi-seed summary ({n_seeds} seeds)")
    print(f"{'='*55}")
    for seed, mean in enumerate(seed_means):
        print(f"  Seed {seed}: {mean:.2f}")
    print(f"  {'─'*45}")
    print(f"  Overall: {overall_mean:.2f} ± {overall_ci:.2f}  (95% CI)")
    solved = "YES" if overall_mean >= 200 else "NO"
    print(f"  Solved (≥200): {solved}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    run_multi_seed(args.config, args.seeds)
