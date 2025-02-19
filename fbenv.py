import gym
from gym import spaces
import numpy as np
import game.wrapped_flappy_bird as game
import cv2

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.game_state = game.GameState()
        self.action_space = spaces.Discrete(1)  # Two actions: do nothing or flap
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 4), dtype=np.uint8)
        self.s_t = None

    def reset(self):
        self.game_state = game.GameState()
        x_t, _, _ = self.game_state.frame_step(np.array([1, 0]))

        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        x_t = self.image_to_frames(x_t)
        return x_t

    def step(self, action):
        actions = np.zeros(2)
        actions[action] = 1
        x_t1, reward, terminal = self.game_state.frame_step(actions)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = self.image_to_frames(x_t1)
        return x_t1, reward, terminal, {}

    def render(self, mode='human'):
        pass

    def image_to_frames(self, x_t):
        if self.s_t is None:
            self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        else:
            x_t = np.reshape(x_t, (80, 80, 1))
            self.s_t = np.append(x_t, self.s_t[:, :, :3], axis=2)
        return self.s_t

    def close(self):
        pass