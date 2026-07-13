##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-15
## File description:
## eval
##

import argparse
import numpy as np
import gymnasium as gym
import torch
import yaml

from baseline import NeuralNetwork, get_end_event, device


def evaluate(model_path, episodes, seed, render):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    online_network = NeuralNetwork(env).to(device)
    online_network.load_state_dict(torch.load(model_path, map_location=device))
    online_network.eval()

    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0.0

        while True:
            state_t = torch.as_tensor(np.array(state), dtype=torch.float32).to(device)
            with torch.no_grad():
                q_values = online_network(state_t)
            action = online_network.predict(q_values)

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                cause = get_end_event(state, terminated, truncated)
                print(f"Episode {episode + 1:4d} | reward: {ep_reward:8.2f} | cause: {cause}")
                rewards.append(ep_reward)
                break

    env.close()
    print(f"\nMean reward over {episodes} episodes: {np.mean(rewards):.2f} Â± {np.std(rewards):.2f}")
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(
        model_path= cfg["model_path"],
        episodes= 100,
        seed= cfg.get("seed", 0),
        render= args.render,
    )