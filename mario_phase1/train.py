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
from mario_phase1.callbacks.checkpoint_callback import CheckpointCallback
from mario_phase1.callbacks.episode_callback import EpisodeCallback
from mario_phase1.ddqn.ddqn_agent import DQNAgent
from mario_phase1.ddqn.ddqn_constraints import DQNAgentConstraints
from mario_phase1.symbolic_components.example_collector import NaiveExampleCollector, ConstraintsExampleCollector
from wrappers.wrappers import apply_wrappers, apply_wrappers
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

logging.initialize(name="T2")


def prepare_config(seed=1):
    return {
        "name": "T2",
        "seed": seed,
        "device": device_name,
        "environment": 'SuperMarioBros-1-1-v0',
        "interval_frequency": 1,
        "checkpoint_frequency": 100000,
        "checkpoint_dir": 'models/',
        "display": True,
        "skip": 4,
        "stack_size": 4,
        "learning_rate": 0.00025,
        "save_replay_buffer": False,
        "detector_model_path": '../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
        "detector_label_path": '../mario_phase0/models/data.yaml',
        "positions_asp": './asp/positions.lp',
        "show_asp": './asp/show.lp',
        "relative_positions_asp": './asp/relative_positions_ext.lp',
        "show_closest_obstacle_asp": './asp/show_closest_obstacle_ext.lp',
        "generate_examples": True,
        "advice_asp": './asp/advice_pipeless_constraints.lp',
        "show_advice_asp": './asp/show_advice.lp',
        "ilasp_binary": './asp/bin/ILASP',
        "ilasp_mode_bias": './asp/ilasp_mode_bias_compact.las',
        "bias": 'negative',
        "constraints": True,
        "forget": True,
        "positive_examples_frequency": 10,
        "symbolic_learn_frequency": 1000,
        "max_induced_programs": 1000
    }


#def run(config, total_time_steps):
def run(config, num_episodes):
    # Setup game
    detector = Detector(config)
    positioner = Positioner(config)
    if config["constraints"]:
        collector = ConstraintsExampleCollector()
    else:
        collector = NaiveExampleCollector()

    inducer = Inducer(config)
    advisor = Advisor(config)

    env = gym_super_mario_bros.make(config["environment"], render_mode='human' if config["display"] else 'rgb',
                                    apply_api_compatibility=True)

    env = apply_wrappers(env, config, detector, positioner, advisor)
    env.reset()

    checkpoint_callback = CheckpointCallback(config)
    interval_callback = IntervalCallback(config["interval_frequency"])
    episode_callback = EpisodeCallback()
    negative_examples_callback = NegativeExampleCallback(collector, offload_freq=config["symbolic_learn_frequency"])
    positive_examples_callback = PositiveExampleCallback(collector, check_freq=1,  # if skip > 0, keep checkfreq 1
                                                         offload_freq=config["symbolic_learn_frequency"])
    induction_callback = InductionCallback(inducer, advisor, check_freq=config["symbolic_learn_frequency"],
                                           max_induced_programs=config["max_induced_programs"],
                                           forget=config["forget"])

    if config["constraints"]:
        agent = DQNAgentConstraints(env,
                                    input_dims=env.observation_space.shape,
                                    num_actions=env.action_space.n,
                                    max_memory_size=4000,
                                    batch_size=16,
                                    gamma=0.90,
                                    lr=config["learning_rate"],
                                    dropout=0.,
                                    exploration_max=1.0,
                                    exploration_min=0.02,
                                    exploration_decay=0.9999961,
                                    pretrained=False,
                                    verbose=1,
                                    seed=config["seed"],
                                    advisor=advisor,
                                    name=config["name"]
                                    )
    else:
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
                         exploration_decay=0.9999961,
                         pretrained=False,
                         verbose=0,
                         seed=config["seed"],
                         advisor=advisor,
                         name=config["name"]
                         )

    # agent.train(min_timesteps_to_train=total_time_steps, callback=[checkpoint_callback,
    #                                                                interval_callback,
    #                                                                episode_callback,
    #                                                                negative_examples_callback,
    #                                                                positive_examples_callback,
    #                                                                induction_callback
    #                                                                ])

    agent.train_episodes(num_episodes=num_episodes, callback=[checkpoint_callback,
                                                              #interval_callback,
                                                              episode_callback,
                                                              #negative_examples_callback,
                                                              #positive_examples_callback,
                                                              #induction_callback
                                                              ])

    env.close()


if __name__ == '__main__':
    #run(prepare_config(seed=1), total_time_steps=100000)
    run(prepare_config(seed=1), num_episodes=5000)
    print("Training done")
