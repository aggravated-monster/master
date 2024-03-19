from collections import deque

import gym
import numpy as np

from gym import Wrapper, ObservationWrapper, ActionWrapper
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from numpy import ndarray
from pandas import DataFrame
# Import Vectorization Wrappers
from stable_baselines3_master.stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from codetiming import Timer

from mario_DQN_baseline.our_logging import our_logging
from mario_DQN_baseline.our_logging.our_logging import Logging


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
    logger = Logging.get_logger('detection')

    def __init__(self, env, detector):
        super().__init__(env)
        self.detector = detector

    @Timer(name="DetectObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation) -> DataFrame:
        positions = self.detector.detect(observation)

        return positions


class PositionObjects(ObservationWrapper):
    logger = Logging.get_logger('positioning')

    def __init__(self, env, positioner):
        super().__init__(env)
        self.positioner = positioner
        # Rationale behind storing the last 3 states:
        # if Mario dies, the callback is only done after the game has already restarted, so the last state is then the first frame of the new game.
        # The one prior to that is the actual final state of the lost game
        # The one prior to that was the state in which the action was chosen that was the cause of death.
        # As it turns out, when falling in holes the situation is a little different, so keeping the last 5 states is probably
        # a Good Plan
        # For positive examples, the story is different, but having 5 states does not hurt
        # TODO Dagmar
        self.relevant_positions = deque(maxlen=10)

    @Timer(name="PositionObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation: DataFrame) -> DataFrame:
        positions = self.positioner.position(observation)
        print(positions)
        # pop oldest if queue full
        # TODO Dagmar
        if len(self.relevant_positions) == self.relevant_positions.maxlen:
            self.relevant_positions.pop()
        # for holes, we have to look back to the oldest observation, so the last action in the lost
        # game has no relationship with this observation anymore at all.
        # Therefore, retrieve the last action from the environment and store it with the observation
        # This is also the action that caused the observation, so semantically, this makes total sense.
        self.relevant_positions.appendleft([None, positions])

        # for now, we don't actually do anything with the result of positioning in the RL chain
        # FWIW: the result of positioning is a list of atoms. If these are going to be used as inputs,
        # an additional wrapper is necessary to convert them to a more useful format
        return observation

# TODO niet nodig
class ChooseAction(ActionWrapper):
    logger = Logging.get_logger('actions')

    def __init__(self, env):
        super().__init__(env)

    @Timer(name="ChooseAction wrapper timer", text="{:0.8f}", logger=logger.info)
    def action(self, act):
        # this is a tad dirty, but it works, and solves the synchronisation problem
        # so keep it. It's not a beauty contest
        # DO convert to ASP though, for convenience
        self.relevant_positions[0][0] = our_logging.RIGHT_ONLY_HUMAN[act] + "."
        #self.relevant_positions.appendleft(our_logging.RIGHT_ONLY_HUMAN[act])

        return act


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
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"])  # Num of frames to apply one action to
    # 3. Detect, position and reduce dimension
    env = DetectObjects(env, detector=detector)  # intercept image and convert to object positions
    env = PositionObjects(env, positioner=positioner)  # intercept image and convert to object positions
    # TODO remove Dagmar
    # env = ChooseAction(env)
    env = TransformAndFlatten(env, dim=config["observation_dim"])
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, config["stack_size"], channels_order='last')

    return env
