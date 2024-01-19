import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info
    

def apply_wrappers(env):
    # 1. Simplify the controls
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=4) # Num of frames to apply one action to
    # 3. Grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')

    return env
