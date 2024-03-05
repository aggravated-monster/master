import numpy as np
from gym import Wrapper, ObservationWrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from numpy import ndarray
from pandas import DataFrame
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


class DetectObjects(ObservationWrapper):
    def __init__(self, env, detector):
        super().__init__(env)
        self.detector = detector

    def observation(self, observation) -> DataFrame:

        positions = self.detector.detect(observation)

        return positions

class PositionObjects(ObservationWrapper):
    def __init__(self, env, positioner):
        super().__init__(env)
        self.positioner = positioner

    def observation(self, observation: DataFrame) -> DataFrame:

        positions = self.positioner.position(observation)

        # for now, we don't actually do anything with the result of positioning in the RL chain
        # FWIW: the result of positioning is a list of atoms. If these are going to be used as inputs,
        # an additional wrapper is necessary to convert them to a more useful format
        return observation


class TransformAndFlatten(ObservationWrapper):
    def __init__(self, env, dim):
        super().__init__(env)
        self.dim = dim

    def observation(self, observation) -> ndarray:
        """Transforms the observation to a ndarray of shape self.dim.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        # expecting input to be a DataFrame with first column 'name'.
        # this column can be dropped. It will only be relevant in Phase2
        positions = observation.drop(['name'], axis=1).to_numpy().copy()

        # make a 1D vector that fits the mlpPolicy
        flattened = positions.reshape(-1)
        # padding the array with negative 1
        padded = np.pad(flattened, (0, self.dim - (positions.shape[0] * positions.shape[1])), 'constant',
                        constant_values=(-1))

        return padded


def apply_wrappers(env, config, detector, positioner):
    # 1. Simplify the controls
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
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
