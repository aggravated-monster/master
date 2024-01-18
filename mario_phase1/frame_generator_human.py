import pygame
import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT
from gym.utils.play import play

from nes_py.wrappers import JoypadSpace
from detector import Detector
from wrappers import apply_wrappers, apply_img_capture_wrappers

import os

from utils import *

import warnings
warnings.filterwarnings('ignore')

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

# if torch.cuda.is_available():
#     print("Using CUDA device:", torch.cuda.get_device_name(0))
# else:
#     print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-3-v0'
# SHOULD_TRAIN = True
DISPLAY = True
# CKPT_SAVE_INTERVAL = 5000
# NUM_OF_EPISODES = 50_000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = apply_img_capture_wrappers(env, ENV_NAME)


mapping = {
    # ...
    () : 0,
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

# #agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
# agent = Agent(input_dims=(4, 4, 16), num_actions=env.action_space.n)
#
# if not SHOULD_TRAIN:
#     folder_name = ""
#     ckpt_name = ""
#     agent.load_model(os.path.join("models", folder_name, ckpt_name))
#     agent.epsilon = 0.2
#     agent.eps_min = 0.0
#     agent.eps_decay = 0.0
#
# env.reset()
# next_state, reward, done, trunc, info = env.step(action=0)
#
# for i in range(NUM_OF_EPISODES):
#     print("Episode:", i)
#     done = False
#     state, _ = env.reset()
#     total_reward = 0
#     while not done:
#         a = agent.choose_action(state)
#         new_state, reward, done, truncated, info  = env.step(a)
#         total_reward += reward
#
#         if SHOULD_TRAIN:
#             agent.store_in_memory(state, a, reward, new_state, done)
#             agent.learn()
#
#         state = new_state
#
#     print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)
#
#     if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
#         agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))
#
#     print("Total reward:", total_reward)
#
# env.close()
