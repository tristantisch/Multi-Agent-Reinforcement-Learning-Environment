import numpy as np
import pygame
from PIL import Image

from LtoS.environments.grid_environments.env_Cleaner import maze
from LtoS.mushroom_rl_extensions.utils.viewer import Viewer


class EnvCleaner(object):
    def __init__(self, N_agent, map_size, seed):
        self.map_size = map_size
        self.seed = seed
        self.occupancy = self.generate_maze(seed)
        self.N_agent = N_agent
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])
        self.viewer = Viewer(self.map_size, self.map_size)
        self.images = []

    def generate_maze(self, seed):
        symbols = {
            # default symbols
            'start': 'S',
            'end': 'X',
            'wall_v': '|',
            'wall_h': '-',
            'wall_c': '+',
            'head': '#',
            'tail': 'o',
            'empty': ' '
        }
        maze_obj = maze.Maze(int((self.map_size - 1) / 2), int((self.map_size - 1) / 2), seed, symbols, 1)
        grid_map = maze_obj.to_np()
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

    def step(self, action_list):
        # TODO: change to per-agent rewards
        reward = np.zeros((self.N_agent,))
        for i in range(len(action_list)):
            if action_list[i] == 0:  # up
                if self.occupancy[self.agt_pos_list[i][0] - 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:  # down
                if self.occupancy[self.agt_pos_list[i][0] + 1][self.agt_pos_list[i][1]] != 1:  # if can move
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:  # left
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] - 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:  # right
                if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1] + 1] != 1:  # if can move
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            if self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] == 2:  # if the spot is dirty
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]] = 0
                reward[i] = reward[i] + 1
        return self.get_global_obs(), reward, np.all(self.occupancy != 2), {}

    def get_global_obs(self):
        # 0: clean, 1: wall, 2: dirty, 3: agent
        # clean/dirty/wall, agents
        obs = np.zeros((self.N_agent, 2, self.map_size, self.map_size))
        for i in range(self.N_agent):
            obs[i, 0, :, :] = self.occupancy  # set clean/dirty/wall
            obs[i, 0][obs[i, 0] == 3] = 0  # set all agent positions to clean
            for i_agent, (pos_x, pos_y) in enumerate(self.agt_pos_list):
                obs[i, 1, pos_x, pos_y] = i_agent + 1
            obs[i, 1, self.agt_pos_list[i][0], self.agt_pos_list[i][1]] = i + 1  # make sure that the agents position is always "on top"
        obs = obs.reshape((self.N_agent, -1))
        agent_index = np.array([[i + 1] for i in range(self.N_agent)])
        obs = np.concatenate((obs, agent_index), axis=1)
        return obs

    def reset(self):
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])

    def render(self):
        def x_y_to_display_x_y(x_orig, y_orig):
            return np.array([y_orig + 0.5, self.map_size - 1 - x_orig + 0.5])

        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:  # clean
                    self.viewer.square(x_y_to_display_x_y(i, j), 0, 0.7, (255, 255, 255))
                if self.occupancy[i][j] == 1:  # wall
                    self.viewer.square(x_y_to_display_x_y(i, j), 0, 0.7, (0, 0, 0))
                if self.occupancy[i][j] == 2:  # dirty
                    self.viewer.square(x_y_to_display_x_y(i, j), 0, 0.7, (0, 255, 0))
                if [i, j] in self.agt_pos_list:  # agent
                    self.viewer.square(x_y_to_display_x_y(i, j), 0, 0.7, (255, 0, 0))
        rgb_array = pygame.surfarray.array3d(self.viewer.screen)
        rgb_array = np.transpose(rgb_array, axes=(1, 0, 2))
        img = Image.fromarray(rgb_array)
        self.images.append(img)
        self.viewer.display(.5)

    def save_gif(self, filepath):
        import imageio
        imageio.mimsave(filepath, self.images[::1], loop=False, duration=400)
        return self.images
