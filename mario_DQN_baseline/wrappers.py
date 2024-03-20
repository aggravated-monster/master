
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation

from stable_baselines3_master.stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env_wrappers.SkipFrame import SkipFrame
from env_wrappers.DetectObjects import DetectObjects
from env_wrappers.TransformAndFlatten import TransformAndFlatten
from env_wrappers.PositionObjects import PositionObjects

def apply_wrappers(env, config):
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"])  # Num of frames to apply one action to
    # 3. convert to greyscale images
    env = GrayScaleObservation(env,keep_dim=True)
    # 4. Wrap inside the Dummy Environment. Standard
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames. Standard
    env = VecFrameStack(env, config["stack_size"], channels_order='last')

    return env

def apply_ASP_wrappers(env, config, detector, positioner):
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"])  # Num of frames to apply one action to
    # 3. Detect, position and reduce dimension
    env = DetectObjects(env, detector=detector)  # intercept image and convert to object positions
    env = PositionObjects(env, positioner=positioner)  # intercept image and convert to object positions
    env = TransformAndFlatten(env, dim=config["observation_dim"])
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, config["stack_size"], channels_order='last')

    return env
