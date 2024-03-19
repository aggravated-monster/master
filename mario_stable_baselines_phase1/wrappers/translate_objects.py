from collections import deque

from codetiming import Timer
from gym import ObservationWrapper
from pandas import DataFrame

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


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
        self.relevant_positions = deque(maxlen=10)

    @Timer(name="PositionObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation: DataFrame) -> DataFrame:
        positions = self.positioner.position(observation)
        print(positions)
        # pop oldest if queue full
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
