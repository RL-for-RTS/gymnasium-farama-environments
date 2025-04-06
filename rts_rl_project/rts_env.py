import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class RTSEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size=10, num_units=3, max_steps=100):
        super(RTSEnv, self).__init__()
        self.grid_size = grid_size
        self.num_units = num_units
        self.max_steps = max_steps
        self.current_step = 0

        # Action space: 4 directions (up, down, left, right) per unit
        self.action_space = spaces.MultiDiscrete([4] * num_units)

        # Observation space: grid with 3 channels (units, resources, enemies)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)

        # Initialize pygame for rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Randomly place units, resources, and enemies
        rng = np.random.RandomState(seed)
        self.unit_positions = rng.randint(0, self.grid_size, size=(self.num_units, 2))
        self.resource_positions = rng.randint(0, self.grid_size, size=(5, 2)).tolist()
        self.enemy_positions = rng.randint(0, self.grid_size, size=(3, 2)).tolist()

        # Update grid
        for pos in self.unit_positions:
            self.grid[pos[0], pos[1], 0] = 1  # Units
        for pos in self.resource_positions:
            self.grid[pos[0], pos[1], 1] = 1  # Resources
        for pos in self.enemy_positions:
            self.grid[pos[0], pos[1], 2] = 1  # Enemies

        return self.grid, {}

    def step(self, action):
        self.current_step += 1
        reward = 0

        # Process actions for each unit
        for unit_idx, move in enumerate(action):
            new_pos = self.unit_positions[unit_idx].copy()
            if move == 0:  # Up
                new_pos[0] = max(0, new_pos[0] - 1)
            elif move == 1:  # Down
                new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
            elif move == 2:  # Left
                new_pos[1] = max(0, new_pos[1] - 1)
            elif move == 3:  # Right
                new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)

            # Update unit position
            self.grid[self.unit_positions[unit_idx][0], self.unit_positions[unit_idx][1], 0] = 0
            self.unit_positions[unit_idx] = new_pos
            self.grid[new_pos[0], new_pos[1], 0] = 1

            # Check for resource collection
            pos_tuple = tuple(new_pos)
            if pos_tuple in [tuple(r) for r in self.resource_positions]:
                reward += 10
                self.grid[new_pos[0], new_pos[1], 1] = 0
                self.resource_positions = [r for r in self.resource_positions if tuple(r) != pos_tuple]

            # Check for enemy collision
            if pos_tuple in [tuple(e) for e in self.enemy_positions]:
                reward -= 5

        # Step penalty
        reward -= 0.1

        # Termination conditions
        terminated = len(self.resource_positions) == 0
        truncated = self.current_step >= self.max_steps

        return self.grid, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if mode == "rgb_array":
            return (self.grid * 255).astype(np.uint8)
        elif mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.grid_size * 40, self.grid_size * 40))
                self.clock = pygame.time.Clock()
            self.screen.fill((255, 255, 255))  # White background
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j, 0]:  # Unit
                        pygame.draw.rect(self.screen, (0, 255, 0), (j * 40, i * 40, 40, 40))
                    if self.grid[i, j, 1]:  # Resource
                        pygame.draw.rect(self.screen, (0, 0, 255), (j * 40, i * 40, 40, 40))
                    if self.grid[i, j, 2]:  # Enemy
                        pygame.draw.rect(self.screen, (255, 0, 0), (j * 40, i * 40, 40, 40))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
