from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gymnasium import spaces
import pygame

class MultiRTSEnv(AECEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, num_agents=2, grid_size=10, max_steps=100):
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32) for agent in self.agents}

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        rng = np.random.RandomState(seed)
        self.agent_positions = {agent: rng.randint(0, self.grid_size, size=2).tolist() for agent in self.agents}
        self.resource_positions = rng.randint(0, self.grid_size, size=(5, 2)).tolist()
        self.enemy_positions = rng.randint(0, self.grid_size, size=(3, 2)).tolist()

        for agent, pos in self.agent_positions.items():
            self.grid[pos[0], pos[1], 0] = 1
        for pos in self.resource_positions:
            self.grid[pos[0], pos[1], 1] = 1
        for pos in self.enemy_positions:
            self.grid[pos[0], pos[1], 2] = 1

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        return {agent: self.grid.copy() for agent in self.agents}

    def step(self, action):
        agent = self.agent_selection
        self.rewards[agent] = 0

        new_pos = self.agent_positions[agent].copy()
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)

        # Update position
        self.grid[self.agent_positions[agent][0], self.agent_positions[agent][1], 0] -= 1
        self.agent_positions[agent] = new_pos
        self.grid[new_pos[0], new_pos[1], 0] += 1

        pos_tuple = tuple(new_pos)
        if pos_tuple in [tuple(r) for r in self.resource_positions]:
            self.rewards[agent] += 10
            self.grid[new_pos[0], new_pos[1], 1] = 0
            self.resource_positions = [r for r in self.resource_positions if tuple(r) != pos_tuple]

        if pos_tuple in [tuple(e) for e in self.enemy_positions]:
            self.rewards[agent] -= 5

        self.rewards[agent] -= 0.1
        self.current_step += 1

        terminated = len(self.resource_positions) == 0
        truncated = self.current_step >= self.max_steps
        self.terminations = {a: terminated for a in self.agents}
        self.truncations = {a: truncated for a in self.agents}

        self.agent_selection = self._agent_selector.next()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return (self.grid * 255).astype(np.uint8)
        elif mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.grid_size * 40, self.grid_size * 40))
                self.clock = pygame.time.Clock()
            self.screen.fill((255, 255, 255))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j, 0]:  # Agents
                        pygame.draw.rect(self.screen, (0, 255, 0), (j * 40, i * 40, 40, 40))
                    if self.grid[i, j, 1]:  # Resources
                        pygame.draw.rect(self.screen, (0, 0, 255), (j * 40, i * 40, 40, 40))
                    if self.grid[i, j, 2]:  # Enemies
                        pygame.draw.rect(self.screen, (255, 0, 0), (j * 40, i * 40, 40, 40))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def observe(self, agent):
        return self.grid.copy()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
