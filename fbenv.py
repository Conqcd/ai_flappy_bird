import gym
from gym import spaces
import numpy as np
import game.wrapped_flappy_bird as game

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.game_state = game.GameState()
        self.action_space = spaces.Discrete(2)  # Two actions: do nothing or flap
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 4), dtype=np.uint8)

    def reset(self):
        self.game_state = game.GameState()
        x_t, _, _ = self.game_state.frame_step(np.array([1, 0]))
        return x_t

    def step(self, action):
        actions = np.zeros(2)
        actions[action] = 1
        x_t1, reward, terminal = self.game_state.frame_step(actions)
        return x_t1, reward, terminal, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass