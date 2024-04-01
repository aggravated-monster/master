import torch
from nes_py.wrappers import JoypadSpace

from mario_phase1.experiments.agents import PositionEnabledAgent, VanillaAgent, DetectionEnabledAgent, \
    ExamplesProducingAgent, PositiveExamplesProducingAgent, NegativeExamplesProducingAgent

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = False
CHECKPOINT_FREQUENCY = 100000
CHECKPOINT_DIR = 'test/'
TENSORBOARD_LOG_DIR = 'logs/tensorboard/'

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'

print(device_name)

config = {
    "device": device_name,
    "skip": 4,
    "stack_size": 4,
    "learning_rate": 0.00025,
    "detector_model_path": '../../mario_phase0/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
    "detector_label_path": '../../mario_phase0/models/data.yaml',
    "positions_asp": '../asp/positions.lp',
    "show_asp": '../asp/show.lp',
    "relative_positions_asp": '../asp/relative_positions.lp',
    "show_closest_obstacle_asp": '../asp/show_closest_obstacle.lp',
    "generate_examples": True,
    "advice_asp": '../asp/advice.lp',
    "show_advice_asp": '../asp/show_advice.lp',
}

NUM_TESTS = 5
NUM_STEPS = 100
START_SEED = 2  # the seed is incremented by 1 in each repetition

negative_examples_producing_agent = NegativeExamplesProducingAgent(config, ENV_NAME, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR, DISPLAY)
negative_examples_producing_agent.execute(NUM_TESTS, NUM_STEPS, START_SEED)

position_enabled_agent = PositionEnabledAgent(config, ENV_NAME, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR, DISPLAY)
position_enabled_agent.execute(NUM_TESTS, NUM_STEPS, START_SEED)

positive_examples_producing_agent = PositiveExamplesProducingAgent(config, ENV_NAME, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR, DISPLAY)
positive_examples_producing_agent.execute(NUM_TESTS, NUM_STEPS, START_SEED)

vanilla_agent = VanillaAgent(config, ENV_NAME, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR, DISPLAY)
vanilla_agent.execute(NUM_TESTS, NUM_STEPS, START_SEED)

detection_enabled_agent = DetectionEnabledAgent(config, ENV_NAME, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR, DISPLAY)
detection_enabled_agent.execute(NUM_TESTS, NUM_STEPS, START_SEED)

examples_producing_agent = ExamplesProducingAgent(config, ENV_NAME, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR, DISPLAY)
examples_producing_agent.execute(NUM_TESTS, NUM_STEPS, START_SEED)

print("Experiment done")
