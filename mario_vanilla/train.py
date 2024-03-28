import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_vanilla.callbacks.checkpoint_callback import CheckpointCallback
from mario_vanilla.callbacks.episode_callback import EpisodeCallback
from mario_vanilla.callbacks.interval_callback import IntervalCallback
from mario_vanilla.ddqn.ddqn import DDQN
from mario_vanilla.vanilla_logging import mario_logging
from wrappers import apply_wrappers

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

LOG_TIMING = True
mario_logging.initialize(LOG_TIMING)

seed = 2

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000
CHECKPOINT_FREQUENCY = 1000
TOTAL_TIME_STEPS = 1000
CHECKPOINT_DIR = 'train/'


config = {
    "device": device_name,
    "skip": 4,
    "stack_size": 4,
    "learning_rate": 0.00025,
    "seed": seed,
}


env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = apply_wrappers(env, config)
env.reset()

checkpointCallback = CheckpointCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, config=config)
intervalCallback = IntervalCallback(check_freq=1)
episodeCallback = EpisodeCallback()

agent = DDQN(env,
             CHECKPOINT_DIR,  # to be moved to callback
             CHECKPOINT_FREQUENCY,  # to be moved to callback
             input_dims=env.observation_space.shape,
             num_actions=env.action_space.n,
             lr=0.00025,
             gamma=0.9,
             epsilon=1.0,
             eps_decay=0.99999975,
             eps_min=0.1,
             replay_buffer_capacity=50000,
             batch_size=32,
             sync_network_rate=10000,
             verbose=1,
             seed=seed,
             device=device)


agent.train(min_timesteps_to_train=TOTAL_TIME_STEPS, callback=[checkpointCallback,
                                                               intervalCallback,
                                                               episodeCallback
                                                               ])


env.close()

print("Training done")