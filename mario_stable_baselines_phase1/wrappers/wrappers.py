from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_stable_baselines_phase1.wrappers.choose_action import ChooseAction
from mario_stable_baselines_phase1.wrappers.detect_objects import DetectObjects
from mario_stable_baselines_phase1.wrappers.skip_frame import SkipFrame
from mario_stable_baselines_phase1.wrappers.transform_and_flatten import TransformAndFlatten
from mario_stable_baselines_phase1.wrappers.translate_objects import PositionObjects
# Import Vectorization Wrappers
from our_stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def apply_wrappers(env, config, detector, positioner, advisor):
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"])  # Num of frames to apply one action to
    # 3. Detect, position and reduce dimension
    env = DetectObjects(env, detector=detector)  # intercept image and convert to object positions
    env = PositionObjects(env, positioner=positioner)  # intercept image and convert to object positions
    env = ChooseAction(env, advisor)
    env = TransformAndFlatten(env, dim=config["observation_dim"])
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, config["stack_size"], channels_order='last')

    return env
