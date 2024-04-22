import gym
import numpy as np


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0