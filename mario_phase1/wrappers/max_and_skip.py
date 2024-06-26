#Code modified from Javier Montalvo (https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning).


import collections

import gym
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        trunc = None
        info = None
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, trunc, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info