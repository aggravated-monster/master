import gym_super_mario_bros
import torch

from ddqn.ddqn import DDQN
from mario_logging import logging
from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.episode_callback import EpisodeCallback
from callbacks.interval_callback import IntervalCallback
from callbacks.negative_example_callback import NegativeExampleCallback
from callbacks.positive_example_callback import PositiveExampleCallback
from callbacks.induction_callback import InductionCallback
from mario_phase1.symbolic_components.example_collector import ExampleCollector
from wrappers.wrappers import apply_wrappers
from symbolic_components.advisor import Advisor
from symbolic_components.detector import Detector
from symbolic_components.positioner import Positioner
from symbolic_components.inducer import Inducer

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

logging.initialize(name="train")


def prepare_config(seed=1):
    return {
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "checkpoint_frequency": 100000,
        "checkpoint_dir": 'models/',
        "display": False,
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025,
        "detector_model_path": '../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
        "detector_label_path": '../mario_phase0/models/data.yaml',
        "positions_asp": './asp/positions.lp',
        "show_asp": './asp/show.lp',
        "relative_positions_asp": './asp/relative_positions.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle.lp',
        "generate_examples": True,
        "show_advice_asp": './asp/show_advice.lp',
        "ilasp_binary": './asp/bin/ILASP',
        "ilasp_mode_bias": './asp/ilasp_mode_bias.las',
        "bias": 'positive',
        "positive_examples_frequency": 10,
        "symbolic_learn_frequency": 1000,
        "max_induced_programs": 100
    }


def run(config, total_time_steps):
    # Setup game
    detector = Detector(config)
    positioner = Positioner(config)
    collector = ExampleCollector()
    inducer = Inducer(config, bias=config['bias'])
    advisor = Advisor(config)

    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    env = apply_wrappers(env, config, detector, positioner, advisor)
    env.reset()

    checkpoint_callback = CheckpointCallback(config)
    interval_callback = IntervalCallback(config["interval_frequency"])
    episode_callback = EpisodeCallback()
    negative_examples_callback = NegativeExampleCallback(collector, offload_freq=config["symbolic_learn_frequency"])
    positive_examples_callback = PositiveExampleCallback(collector, check_freq=1, # if skip > 0, keep checkfreq 1
                                                         offload_freq=config["symbolic_learn_frequency"])
    induction_callback = InductionCallback(inducer, advisor, check_freq=config["symbolic_learn_frequency"],
                                           max_induced_programs=config["max_induced_programs"])

    agent = DDQN(env,
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
                 seed=config["seed"],
                 device=device)

    agent.train(min_timesteps_to_train=total_time_steps, callback=[checkpoint_callback,
                                                                   interval_callback,
                                                                   episode_callback,
                                                                   negative_examples_callback,
                                                                   positive_examples_callback,
                                                                   induction_callback
                                                                   ])

    env.close()


if __name__ == '__main__':
    run(prepare_config(seed=42),
        total_time_steps=1500)
    print("Training done")
