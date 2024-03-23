from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_stable_baselines_phase1.wrappers.choose_action import ChooseAction
from mario_stable_baselines_phase1.wrappers.detect_objects import DetectObjects
from mario_stable_baselines_phase1.wrappers.skip_frame import SkipFrame
from mario_stable_baselines_phase1.wrappers.transform_and_flatten import TransformAndFlatten
from mario_stable_baselines_phase1.wrappers.translate_objects import PositionObjects
# Import Vectorization Wrappers
from master_stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def apply_wrappers(env, config, detector, po100sitioner, advisor):
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"])  # Num of frames to apply one action to
    # The following set of wrappers do not change the observation (it will always be raw pixels)
    # but they use the raw pixel values to perform a series of symbolic transformations on them
    # 3a. Detect objects and store them for later use
    #env = DetectObjects(env, detector=detector)  # intercept image and convert to object positions
    # 3b. Translate the bounding boxes to an object/relational representation
    #env = PositionObjects(env, positioner=positioner)  # intercept image and convert to object positions
    # 3c. Invoke the Advisor
    #env = ChooseAction(env, advisor)
    # From here on, the observation IS altered again, for efficiency purposes in the RL environment
    env = ResizeObservation(env, shape=84)
    # 4. Grayscale; the cnn inside the DQN is perfectly capable of handling grayscale images
    env = GrayScaleObservation(env, keep_dim=True)
    # 5. Wrap inside the Dummy Environment. Standard.
    env = DummyVecEnv([lambda: env])
    # 6. Stack the frames. Standard.
    env = VecFrameStack(env, config["stack_size"], channels_order='last')

    return env
