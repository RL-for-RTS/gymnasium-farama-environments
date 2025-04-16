import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from multi_rts_env import MultiRTSEnv

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Adjusted for (3, 10, 10) input
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def batchify_obs(obs, device):
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, 3, 1, 2)  # (batch, channels, height, width)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    return obs

def batchify(x, device):
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x

def unbatchify(x, env):
    x = x.cpu().numpy()
    return {a: x[i] for i, a in enumerate(env.agents)}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    max_cycles = 100
    total_episodes = 10

    env = MultiRTSEnv(num_agents=2, max_steps=max_cycles)
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n

    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    rb_obs = torch.zeros((max_cycles, num_agents, 3, 10, 10)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    for episode in range(total_episodes):
        next_obs = env.reset()
        total_episodic_return = np.zeros(num_agents)
        step = 0

        with torch.no_grad():
            while env.agents and step < max_cycles:
                for agent in env.agents:
                    obs = batchify_obs({agent: next_obs[agent]}, device)
                    action, logprob, _, value = agent.get_action_and_value(obs)
                    env.step(action.item())
                    next_obs = {a: env.observe(a) for a in env.agents}
                    rb_obs[step, env.agents.index(agent)] = obs.squeeze(0)
                    rb_actions[step, env.agents.index(agent)] = action
                    rb_logprobs[step, env.agents.index(agent)] = logprob
                    rb_rewards[step, env.agents.index(agent)] = env.rewards[agent]
                    rb_terms[step, env.agents.index(agent)] = env.terminations[agent]
                    rb_values[step, env.agents.index(agent)] = value.flatten()
                total_episodic_return += rb_rewards[step].cpu().numpy()
                step += 1
                env.render()

        end_step = step
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = rb_rewards[t] + gamma * rb_values[t + 1] * (1 - rb_terms[t + 1]) - rb_values[t]
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                end = start + batch_size
                batch_index = b_index[start:end]
                _, newlogprob, entropy, value = agent.get_action_and_value(b_obs[batch_index], b_actions.long()[batch_index])
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(value - b_values[batch_index], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var}")
        print("\n-------------------------------------------\n")

    env.close()
