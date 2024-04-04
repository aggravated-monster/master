from collections import deque

from codetiming import Timer
from gym import ObservationWrapper
from pandas import DataFrame

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class PositionObjects(ObservationWrapper):
    logger = Logging.get_logger('positioning')

    def __init__(self, env, positioner, seed):
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
        self.seed = seed

    #@Timer(name="PositionObjects wrapper timer", text="{:0.8f}", logger=logger.info)
    def observation(self, observation):

        # This wrapper twins with the track_action wrapper, and first stores the
        # relevant positions, for later use.

        # keep track of current observation (and add the action later once it is available),
        # as they are used to create examples.
        # this is a tad dirty, but it works, and solves the synchronisation problem
        # so keep it. It's not a beauty contest
        # DO convert to ASP though, for convenience
        # We are not interested in the positions and actions chosen while Mario is in a state
        # where choosing actions does not change his course.
        # This happens in-flight.
        # To make things easier, we will only record actions and positions that take place at ground-level
        # This should be sufficient to encounter all different scenarios that can be generalised over
        # not-ground-level positions
        # Mario's y-pos can be obtained from the SuperMarioBrosEnv, in protected property _y_position
        # This is terribly ugly, but given time constraints, we cannot redesign the environment
        if self.unwrapped.env._y_position < 80:

            text = str(self.seed) + ";{:0.8f}"

            with Timer(name="Translate object wrapper timer", text=text, logger=self.logger.info):

                # retrieve the detected objects from the environment
                detected_objects = self.env.detected_objects
                if detected_objects is not None:
                    positions = self.positioner.position(detected_objects)
                    # pop oldest if queue full
                    if len(self.relevant_positions) == self.relevant_positions.maxlen:
                        self.relevant_positions.pop()
                    # for holes, we have to look back to the oldest observation, so the last action in the lost
                    # game has no relationship with this observation anymore at all.
                    # Therefore, retrieve the last action from the environment and store it with the observation
                    # This is also the action that caused the observation, so semantically, this makes total sense.
                    # We do not have access to the last action at this point, but the first argument
                    # (here None as a placeholder) will be filled in the action wrapper that succeeds this wrapper.
                    self.relevant_positions.appendleft([None, positions])

                # for now, we don't actually do anything with the result of positioning in the RL chain
                # FWIW: the result of positioning is a list of atoms. If these are going to be used as inputs,
                # an additional wrapper is necessary to convert them to a more useful format

        # return the observation untouched
        return observation
