import pygame

import gym_super_mario_bros
from gym import ObservationWrapper
from gym.error import DependencyNotInstalled
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.utils.play import play

from nes_py.wrappers import JoypadSpace

import os

from mario_phase0.utils import *

import warnings
warnings.filterwarnings('ignore')


class CaptureFrames(ObservationWrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.env_name = env_name

    def observation(self, observation):
        try:
            import cv2
        except ImportError:
            raise DependencyNotInstalled(
                "opencv is not installed, run 'pip install gym[other]'")
        cv2.imshow('game', observation)
        cv2.imwrite('./frames/img_' + self.env_name + '_' + get_current_date_time_string() + '.png', observation)


def apply_img_capture_wrappers(env, env_name):
    env = CaptureFrames(env, env_name)  # intercept image and convert to object positions

    return env

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

ENV_NAME = 'SuperMarioBros-1-1-v0'

DISPLAY = True

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = apply_img_capture_wrappers(env, ENV_NAME)


mapping = {
    # ...
    (): 0,
    (pygame.K_f,): 1,
    (pygame.K_f, pygame.K_i): 2,
    (pygame.K_f, pygame.K_j): 3,
    (pygame.K_f, pygame.K_i, pygame.K_j): 4,
    (pygame.K_i, ): 5,
    (pygame.K_d,): 6,
    (pygame.K_d, pygame.K_i): 7,
    (pygame.K_d, pygame.K_j): 8,
    (pygame.K_d, pygame.K_j, pygame.K_i): 9,
    (pygame.K_x,): 10,
    (pygame.K_e,): 11

    # ...
}

play(env, keys_to_action=mapping)


