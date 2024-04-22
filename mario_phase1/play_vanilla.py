import os

import gym_super_mario_bros
import torch

from callbacks.episode_callback import EpisodeCallback
from mario_logging import logging
from ddqn.ddqn_vanilla import DDQNVanilla
from wrappers.wrappers import apply_wrappers_vanilla

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

logging.initialize()


def prepare_config(seed=1):
    return {
        "seed": seed,
        "device": device_name,
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025
    }


def run(config, num_episodes):
    # Setup game
    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    env = apply_wrappers_vanilla(env, config)
    env.reset()

    episode_callback = EpisodeCallback()

    agent = DDQNVanilla(env,
                        input_dims=env.observation_space.shape,
                        num_actions=env.num_actions.n,
                        lr=0.00025,
                        gamma=0.9,
                        epsilon=1.0,
                        eps_decay=0.99999975,
                        eps_min=0.1,
                        replay_buffer_capacity=50000,
                        batch_size=32,
                        sync_network_rate=10000,
                        verbose=1,
                        seed=config["seed"],
                        device=device)

    agent.load_model(path=os.path.join("./train", "model_8000000_iter.pt"))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

    agent.play(num_episodes, callback=episode_callback)

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=51),
        num_episodes=1000)
    print("Game done")
