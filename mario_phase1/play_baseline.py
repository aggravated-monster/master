import os

import gym_super_mario_bros
import torch

from mario_logging import logging
from mario_phase1.callbacks.episode_callback_alt import EpisodeCallbackAlt
from mario_phase1.ddqn.ddqn_agent import DQNAgent
from wrappers.wrappers import apply_wrappers_baseline

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

logging.initialize(name="play-baseline")


def prepare_config(seed=1):
    return {
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025,
        "display": True,
    }


def run(config, num_episodes):
    # Setup game
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    env = apply_wrappers_baseline(env, config)
    env.reset()

    episode_callback = EpisodeCallbackAlt()

    agent = DQNAgent(env,
                     input_dims=env.observation_space.shape,
                     num_actions=env.action_space.n,
                     max_memory_size=4000,
                     batch_size=16,
                     gamma=0.90,
                     lr=config["learning_rate"],
                     dropout=0.,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     pretrained=False,
                     verbose=0,
                     seed=config["seed"]
                     )

    agent.load_model(path=os.path.join("./models_baseline", "1000000_1_target_net.pt"))
    agent.epsilon = 0.02
    agent.exploration_min = 0.0
    agent.exploration_decay = 0.0

    agent.play(num_episodes, callback=episode_callback)

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=1),
        num_episodes=1000)
    print("Game done")
