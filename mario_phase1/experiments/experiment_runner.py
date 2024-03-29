import gym_super_mario_bros
import torch
from nes_py.wrappers import JoypadSpace

from mario_phase1.experiments.agents import PositionEnabledAgent, VanillaAgent

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

# Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)

position_enabled_agent = PositionEnabledAgent(config, env, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR )
position_enabled_agent.execute(5, 100)

vanilla_agent = VanillaAgent(config, env, device, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR )
vanilla_agent.execute(5, 100)

print("Experiment done")
