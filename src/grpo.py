import torch
from torch import nn
from torch.distributions import Categorical

class GRPO:
    """Group Relative Policy Optimization.

    This implementation avoids a separate value function. For each
    observation, multiple actions are sampled from the policy and scored
    with a reward model. The mean score of the group serves as a baseline
    so updates remain relative. The approach keeps memory use low while
    still allowing gradient based optimisation.
    """

    def __init__(self, env, reward_model, hidden_size=64, group_size=4, lr=1e-3, seed=42):
        self.env = env
        self.reward_model = reward_model
        self.group_size = group_size
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        torch.manual_seed(seed)
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions)
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        logits = self.policy(obs_tensor)
        if deterministic:
            action = logits.argmax(dim=-1).item()
        else:
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        return action, None

    def _sample_group(self, obs):
        actions = []
        log_probs = []
        dist = Categorical(logits=self.policy(torch.tensor(obs, dtype=torch.float32)))
        for _ in range(self.group_size):
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))
        return torch.stack(actions), torch.stack(log_probs)

    def learn(self, total_timesteps=1000):
        obs = self.env.reset()
        for _ in range(total_timesteps):
            actions, log_probs = self._sample_group(obs)
            rewards = torch.tensor([self.reward_model(obs, a.item()) for a in actions], dtype=torch.float32)
            baseline = rewards.mean()
            loss = -((rewards - baseline) * log_probs).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            obs, _, done, _ = self.env.step(actions[0].item())
            if done:
                obs = self.env.reset()
