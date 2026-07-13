##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-15
## File description:
## baseline
##

import argparse
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import yaml
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_policy(_obs):
    return random.randint(0, 3)

def heuristic(obs):
    _, _, _, vy, angle, angular_vel, left_leg, right_leg = obs

    if left_leg and right_leg:
        return 0
    tilt = angle + 0.5 * angular_vel
    if tilt > 0.15:
        return 3
    if tilt < -0.15:
        return 1
    if vy < -1.0:
        return 2
    return 0

def heuristic_policy(obs):
    return heuristic(obs)


class NeuralNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.linear1 = nn.Linear(env.observation_space.shape[0], 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def predict(self, x):
        return x.argmax().detach().item()


class Memory():
    def __init__(self, capacity, batch_size):
        self.D = deque(maxlen=capacity)
        self.batch_size = batch_size

    def store_transition(self, state, action, reward, done, new_state):
        return self.D.append((state, action, reward, done, new_state))

    def retrieve_transitions(self):
        transitions = random.sample(self.D, self.batch_size)

        states = ([t[0] for t in transitions])
        actions = ([t[1] for t in transitions])
        rewards = ([t[2] for t in transitions])
        dones = ([t[3] for t in transitions])
        new_states = ([t[4] for t in transitions])

        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)
        new_states_t = torch.FloatTensor(new_states).to(device)

        return states_t, actions_t, rewards_t, dones_t, new_states_t
    def get_action(self, epsilon, greedy_action, action_space):
        random_value = random.random()
        if random_value > epsilon:
            return greedy_action
        if random_value <= epsilon:
            return action_space.sample()

def DQN(epsilon, online_network, target_network, loss_fn, optimizer, env, state, steps,
        replay, gamma, frames, update_frequency):
    episode_reward = 0.0
    episode_loss = []

    for _ in range(frames):
        epsilon = max(0.01, epsilon * 0.99995)
        # interacting with the environment
        state_t = torch.as_tensor(np.array(state), dtype=torch.float32).to(device)
        q_values = online_network.forward(state_t)
        greedy_action = online_network.predict(q_values)
        action = replay.get_action(epsilon, greedy_action, env.action_space)

        new_state, reward, done, trunc, _ = env.step(action)
        episode_reward += reward
        replay.store_transition(state, action, reward, done, new_state)

        state = new_state
        ## updating network
        states, actions, rewards, dones, new_states = replay.retrieve_transitions()

        max_next_Q = target_network.forward(new_states).max(dim=1, keepdim=True)[0]
        Y = rewards + gamma * (1 - dones) * max_next_Q

        current_Q = online_network.forward(states).gather(dim=1, index=actions)

        loss = loss_fn(Y, current_Q)
        episode_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if steps % update_frequency == 0:
            target_network.load_state_dict(online_network.state_dict())
        steps += 1
        if done or trunc:
            break

    cause = get_end_event(state, done, trunc)
    return episode_reward, episode_loss, epsilon, steps, cause

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedMemory:
    def __init__(self, capacity, batch_size, alpha=0.6, beta_start=0.4, beta_frames=50000):
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = 1e-5
        self.max_priority = 1.0
        self.frame = 1

    def store_transition(self, state, action, reward, done, new_state):
        self.tree.add(self.max_priority, (state, action, reward, done, new_state))

    def retrieve_transitions(self):
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        segment = self.tree.total() / self.batch_size
        priorities, transitions, indices = [], [], []

        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            transitions.append(data)
            indices.append(idx)

        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= weights.max()

        states_t = torch.FloatTensor([t[0] for t in transitions]).to(device)
        actions_t = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor([t[2] for t in transitions]).unsqueeze(1).to(device)
        dones_t = torch.FloatTensor([t[3] for t in transitions]).unsqueeze(1).to(device)
        new_states_t = torch.FloatTensor([t[4] for t in transitions]).to(device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(device)

        return states_t, actions_t, rewards_t, dones_t, new_states_t, weights_t, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def get_action(self, epsilon, greedy_action, action_space):
        if random.random() > epsilon:
            return greedy_action
        return action_space.sample()

def DoubleDQN(epsilon, online_network, target_network, loss_fn, optimizer, env, state, steps,
              replay, gamma, frames, update_frequency):
    episode_reward = 0.0
    episode_loss = []

    for _ in range(frames):
        epsilon = max(0.01, epsilon * 0.99995)
        state_t = torch.as_tensor(np.array(state), dtype=torch.float32).to(device)
        q_values = online_network.forward(state_t)
        greedy_action = online_network.predict(q_values)
        action = replay.get_action(epsilon, greedy_action, env.action_space)

        new_state, reward, done, trunc, _ = env.step(action)
        episode_reward += reward
        replay.store_transition(state, action, reward, done, new_state)

        state = new_state
        states, actions, rewards, dones, new_states = replay.retrieve_transitions()

        best_actions = online_network.forward(new_states).argmax(dim=1, keepdim=True)
        next_Q = target_network.forward(new_states).gather(dim=1, index=best_actions)
        Y = rewards + gamma * (1 - dones) * next_Q

        current_Q = online_network.forward(states).gather(dim=1, index=actions)

        loss = loss_fn(Y, current_Q)
        episode_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if steps % update_frequency == 0:
            target_network.load_state_dict(online_network.state_dict())
        steps += 1
        if done or trunc:
            break

    cause = get_end_event(state, done, trunc)
    return episode_reward, episode_loss, epsilon, steps, cause


def DoubleDQNPER(epsilon, online_network, target_network, loss_fn, optimizer, env, state, steps,
                 replay, gamma, frames, update_frequency):
    episode_reward = 0.0
    episode_loss = []

    for _ in range(frames):
        epsilon = max(0.01, epsilon * 0.99995)
        state_t = torch.as_tensor(np.array(state), dtype=torch.float32).to(device)
        q_values = online_network.forward(state_t)
        greedy_action = online_network.predict(q_values)
        action = replay.get_action(epsilon, greedy_action, env.action_space)

        new_state, reward, done, trunc, _ = env.step(action)
        episode_reward += reward
        replay.store_transition(state, action, reward, done, new_state)

        state = new_state
        states, actions, rewards, dones, new_states, weights, indices = replay.retrieve_transitions()

        best_actions = online_network.forward(new_states).argmax(dim=1, keepdim=True)
        next_Q = target_network.forward(new_states).gather(dim=1, index=best_actions)
        Y = rewards + gamma * (1 - dones) * next_Q

        current_Q = online_network.forward(states).gather(dim=1, index=actions)

        td_errors = (Y - current_Q).detach()
        loss = (weights * F.smooth_l1_loss(current_Q, Y.detach(), reduction='none')).mean()
        episode_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        replay.update_priorities(indices, td_errors.squeeze().abs().cpu().numpy())

        if steps % update_frequency == 0:
            target_network.load_state_dict(online_network.state_dict())
        steps += 1
        if done or trunc:
            break

    cause = get_end_event(state, done, trunc)
    return episode_reward, episode_loss, epsilon, steps, cause


POLICIES = {
    "random": random_policy,
    "heuristic": heuristic_policy,
    "DQN": DQN,
    "DoubleDQN": DoubleDQN,
    "DoubleDQNPER": DoubleDQNPER,
}

def get_end_event(obs, terminated, truncated):
    if truncated:
        return "sleep"
    if terminated:
        if obs[6] and obs[7]:
            return "landed"
        if abs(obs[0]) >= 0.95:
            return "out_of_view"
        return "crash"
    return "quit"

def save_plots(cfg, train_rewards, train_epsilons, train_losses, train_causes, train_lengths):
    plt.switch_backend('Agg')
    policy_name = cfg["policy"]
    is_rl = policy_name in ("DQN", "DoubleDQN", "DoubleDQNPER")
    plot_prefix = os.path.splitext(cfg.get("model_path", f"runs/{policy_name}"))[0]
    os.makedirs(os.path.dirname(plot_prefix) or ".", exist_ok=True)

    episodes = list(range(1, len(train_rewards) + 1))
    window = max(1, min(50, len(train_rewards) // 5))
    rolling = np.convolve(train_rewards, np.ones(window) / window, mode='valid')

    ncols = 2 if is_rl else 2
    nrows = 2 if is_rl else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    axes = np.array(axes).flatten()

    ax = axes[0]
    ax.plot(episodes, train_rewards, alpha=0.35, label="reward")
    ax.plot(range(window, len(train_rewards) + 1), rolling, label=f"{window}-ep mean")
    ax.axhline(200, color="green", linestyle="--", linewidth=1, label="solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Returns")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(episodes, train_lengths, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode length")

    if is_rl:
        ax = axes[2]
        ax.plot(episodes, train_epsilons, color="darkorange")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("Epsilon decay")

        ax = axes[3]
        ax.plot(episodes, train_losses, alpha=0.5, color="purple")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Mean loss per episode")

    fig.suptitle(f"{policy_name} — training")
    fig.tight_layout()
    fig.savefig(f"{plot_prefix}_training.png", dpi=100)
    plt.close(fig)

    cause_labels = ["landed", "crash", "sleep", "out_of_view", "quit"]
    cause_colors = {"landed": "#4caf50", "crash": "#f44336", "sleep": "#ff9800",
                    "out_of_view": "#9e9e9e", "quit": "#90caf9"}
    counts = {c: train_causes.count(c) for c in cause_labels if train_causes.count(c) > 0}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.keys(), counts.values(), color=[cause_colors[c] for c in counts])
    ax.set_xlabel("Cause")
    ax.set_ylabel("Count")
    ax.set_title(f"{policy_name} — termination causes")
    fig.tight_layout()
    fig.savefig(f"{plot_prefix}_causes.png", dpi=100)
    plt.close(fig)

    print(f"Plots saved to {plot_prefix}_training.png and {plot_prefix}_causes.png")


def run(cfg, render=False):
    policy_name = cfg["policy"]
    learning_rate = cfg["learning_rate"]
    gamma = cfg["gamma"]
    batch_size = cfg["batch_size"]
    memory_capacity = cfg["memory_capacity"]
    frames = cfg["frames"]
    update_frequency = cfg["update_frequency"]

    video_interval = cfg.get("video_interval", 10)
    video_dir = os.path.splitext(cfg.get("model_path", f"runs/{policy_name}"))[0] + "_videos"

    if render:
        env = gym.make("LunarLander-v3", render_mode="human")
    else:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep, n=video_interval: ep % n == 0,
            name_prefix=policy_name,
            disable_logger=True,
        )
        print(f"Videos will be saved to {video_dir}/ every {video_interval} episodes")

    online_network = NeuralNetwork(env).to(device)
    target_network = NeuralNetwork(env).to(device)
    target_network.load_state_dict(online_network.state_dict())
    optimizer = torch.optim.RMSprop(params=online_network.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    if policy_name == "DoubleDQNPER":
        replay = PrioritizedMemory(memory_capacity, batch_size)
    else:
        replay = Memory(memory_capacity, batch_size)

    epsilon = 1.0
    steps = 0
    train_rewards = []
    train_epsilons = []
    train_losses = []
    train_causes = []
    train_lengths = []

    if policy_name in ("DQN", "DoubleDQN", "DoubleDQNPER"):
        state, _ = env.reset(seed=cfg["seed"])
        for _ in range(memory_capacity):
            action = env.action_space.sample()

            new_state, reward, done, _, _ = env.step(action)

            replay.store_transition(state, action, reward, done, new_state)

            state = new_state

            if done:
                state, _ = env.reset()
    for episode in range(cfg["episodes"]):
        obs, _ = env.reset(seed=cfg["seed"] + episode)

        if policy_name in ("DQN", "DoubleDQN", "DoubleDQNPER"):
            step_fn = POLICIES[policy_name]
            steps_before = steps
            ep_reward, ep_loss, epsilon, steps, cause = step_fn(
                epsilon, online_network, target_network, loss_fn, optimizer, env, obs, steps,
                replay, gamma, frames, update_frequency
            )
            mean_loss = np.mean(ep_loss) if ep_loss else 0.0
            train_epsilons.append(epsilon)
            train_losses.append(mean_loss)
            train_lengths.append(steps - steps_before)
            print(f"Episode {episode + 1:4d} | reward: {ep_reward:8.2f} | loss: {mean_loss:.4f} | eps: {epsilon:.3f} | cause: {cause}")
        else:
            ep_reward = 0.0
            ep_length = 0
            terminated = truncated = False
            policy = POLICIES[policy_name]
            while True:
                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_length += 1
                if terminated or truncated:
                    break
            cause = get_end_event(obs, terminated, truncated)
            train_lengths.append(ep_length)
            print(f"Episode {episode + 1:4d} | reward: {ep_reward:8.2f} | cause: {cause}")

        train_rewards.append(ep_reward)
        train_causes.append(cause)

    env.close()

    if policy_name in ("DQN", "DoubleDQN", "DoubleDQNPER"):
        model_path = cfg.get("model_path", "model.pth")
        torch.save(online_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    save_plots(cfg, train_rewards, train_epsilons, train_losses, train_causes, train_lengths)
    return train_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-interval", type=int, default=100,
                        help="Record a video every N episodes (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.video_interval is not None:
        cfg["video_interval"] = args.video_interval

    run(cfg, render=args.render)
