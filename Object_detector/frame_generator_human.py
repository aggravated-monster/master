import pygame

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.utils.play import play

from nes_py.wrappers import JoypadSpace
from to_delete.mario_phase1_youtube_vanilla.wrappers import apply_img_capture_wrappers

import os

from to_delete.mario_phase1_youtube_vanilla.utils import *

import warnings
warnings.filterwarnings('ignore')

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

ENV_NAME = 'SuperMarioBros-3-1-v0'

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


