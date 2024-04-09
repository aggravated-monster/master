import gym_super_mario_bros
import torch

from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.episode_callback import EpisodeCallback
from callbacks.interval_callback import IntervalCallback
from ddqn.ddqn import DDQN
from mario_logging import logging
from callbacks.negative_example_callback import NegativeExampleCallback
from callbacks.positive_example_callback import PositiveExampleCallback
from mario_phase1.callbacks.induction_callback import InductionCallback
from mario_phase1.symbolic_components.inducer import Inducer
from mario_phase1.wrappers.wrappers import apply_wrappers
from symbolic_components.advisor import Advisor
from symbolic_components.detector import Detector
from symbolic_components.positioner import Positioner

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'
else:
    print("CUDA is not available")

LOG_TIMING = True
logging.initialize(LOG_TIMING)

seed = 51

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = False

CHECKPOINT_FREQUENCY = 100000
TOTAL_TIME_STEPS = 100000
SYMBOLIC_LEARN_FREQUENCY = 50000
CHECKPOINT_DIR = 'train/'


config = {
    "device": device_name,
    "skip": 4,
    "stack_size": 4,
    "learning_rate": 0.00025,
    "seed": seed,
    "detector_model_path": '../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
    "detector_label_path": '../mario_phase0/models/data.yaml',
    "positions_asp": './asp/positions.lp',
    "show_asp": './asp/show.lp',
    "relative_positions_asp": './asp/relative_positions.lp',
    "show_closest_obstacle_asp": './asp/show_closest_obstacle.lp',
    "generate_examples": True,
    "show_advice_asp": './asp/show_advice.lp',
    "ilasp_binary": './asp/bin/ILASP',
    'ilasp_background_searchspace': './asp/ilasp_background_searchspace.las',
    'bias': 'positive'
}

# Setup game
# 1. Create the object detector. This is a YOLO8 model
detector = Detector(config)
# 2. Create the Translator
positioner = Positioner(config)
# 3. Create the Inducer
inducer = Inducer(config, bias=config['bias'])
# 4. Create the Advisor
advisor = Advisor(config)



env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)

env = apply_wrappers(env, config, detector, positioner, advisor, seed)
env.reset()

checkpointCallback = CheckpointCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, config=config)
intervalCallback = IntervalCallback(check_freq=1)
episodeCallback = EpisodeCallback()
negativeExamplesCallback = NegativeExampleCallback(offload_freq=SYMBOLIC_LEARN_FREQUENCY)
positiveExamplesCallback = PositiveExampleCallback(check_freq=1, offload_freq=SYMBOLIC_LEARN_FREQUENCY)
inductionCallback = InductionCallback(inducer, advisor, check_freq=SYMBOLIC_LEARN_FREQUENCY, max_induced_programs=10)

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
             seed=seed,
             device=device)


agent.train(min_timesteps_to_train=TOTAL_TIME_STEPS, callback=[checkpointCallback,
                                                               intervalCallback,
                                                               episodeCallback,
                                                               negativeExamplesCallback,
                                                               positiveExamplesCallback,
                                                               inductionCallback
                                                               ])


env.close()

print("Training done")
