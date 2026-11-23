import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLinkArmEnv:
    """Ambiente simples de braço robótico planar de 2 juntas (ações contínuas em [-1, 1])."""

    def __init__(self, max_steps: int = 100, dt: float = 0.05):
        self.max_steps = max_steps
        self.dt = dt
        self.link_lengths = np.array([1.0, 0.8], dtype=np.float32)
        self.state_dim = 6  # [q1, q2, dq1, dq2, target_x, target_y]
        self.action_dim = 2  # [torque_1, torque_2] normalizados em [-1, 1]
        self.max_torque = 2.0
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.steps = 0

    def _end_effector_pos(self, q1: float, q2: float) -> Tuple[float, float]:
        l1, l2 = self.link_lengths
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        return float(x), float(y)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        q = rng.uniform(-np.pi, np.pi, size=2).astype(np.float32)
        dq = np.zeros(2, dtype=np.float32)
        # Alvo dentro do alcance máximo (1.8) e mínimo (0.2)
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.3, 1.6)
        target = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
        )
        self.state = np.concatenate([q, dq, target]).astype(np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        self.steps += 1
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        torques = clipped_action * self.max_torque

        q1, q2, dq1, dq2, tx, ty = self.state
        damping = 0.25

        # Dinâmica simplificada de segunda ordem com amortecimento
        ddq1 = torques[0] - damping * dq1
        ddq2 = torques[1] - damping * dq2

        dq1 = float(np.clip(dq1 + ddq1 * self.dt, -5.0, 5.0))
        dq2 = float(np.clip(dq2 + ddq2 * self.dt, -5.0, 5.0))
        q1 = float(((q1 + dq1 * self.dt + np.pi) % (2 * np.pi)) - np.pi)
        q2 = float(((q2 + dq2 * self.dt + np.pi) % (2 * np.pi)) - np.pi)

        ex, ey = self._end_effector_pos(q1, q2)
        dist = float(np.hypot(tx - ex, ty - ey))

        # Recompensa negativa pela distância ao alvo + pequena penalidade de esforço
        effort_penalty = 0.01 * float(np.sum(clipped_action**2))
        reward = -dist - effort_penalty
        if dist < 0.08:
            reward += 2.0

        done = self.steps >= self.max_steps
        self.state = np.array([q1, q2, dq1, dq2, tx, ty], dtype=np.float32)
        return self.state.copy(), reward, done


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 50_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Saída contínua em [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Duas redes Q (Q1 e Q2) para reduzir superestimação no TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=1))


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    exploration_noise: float = 0.15
    batch_size: int = 128
    lr: float = 1e-3


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, cfg: TD3Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def train_step(self, buffer: ReplayBuffer) -> None:
        if buffer.size < self.cfg.batch_size:
            return

        self.total_it += 1
        state, action, reward, next_state, done = buffer.sample(
            self.cfg.batch_size, self.device
        )

        with torch.no_grad():
            # 1. Target policy smoothing (ruído limitado na ação alvo)
            noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            # 2. Clipped double Q-learning (mínimo entre Q1 e Q2 alvo)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.cfg.gamma * target_q

        # Atualiza redes críticas
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 3. Atualização atrasada do ator e das redes-alvo
        if self.total_it % self.cfg.policy_freq == 0:
            actor_loss = -self.critic.q1_only(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
                )


def main() -> None:
    rng = np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TwoLinkArmEnv(max_steps=80)
    cfg = TD3Config()
    agent = TD3Agent(env.state_dim, env.action_dim, cfg, device)
    buffer = ReplayBuffer(env.state_dim, env.action_dim)

    episodes = 120
    warmup_steps = 500
    step_count = 0

    for ep in range(1, episodes + 1):
        state = env.reset(rng)
        ep_reward = 0.0
        done = False

        while not done:
            step_count += 1
            if step_count < warmup_steps:
                action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, noise_std=cfg.exploration_noise)

            next_state, reward, done = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            agent.train_step(buffer)

            state = next_state
            ep_reward += reward

        if ep % 20 == 0 or ep == 1:
            print(f"Episódio {ep:3d}/{episodes} | Recompensa acumulada: {ep_reward:7.2f}")

    # Avaliação final sem ruído de exploração
    eval_state = env.reset(rng)
    eval_action = agent.select_action(eval_state, noise_std=0.0)
    print(f"\nAção prevista pelo ator para o estado inicial: {eval_action}")


if __name__ == "__main__":
    main()