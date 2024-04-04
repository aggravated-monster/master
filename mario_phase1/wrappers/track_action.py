from abc import ABC

import numpy as np
from codetiming import Timer
from gym import ActionWrapper

from mario_stable_baselines_phase1.our_logging import our_logging
from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class TrackAction(ActionWrapper, ABC):

    def __init__(self, env):
        super().__init__(env)

    def action(self, act):

        # This wrapper twins with the translate_objects wrapper, and adds the action to the
        # relevant positions, for later use.

        # keep track of the action taken, together with the current observation,
        # as they are used to create examples.
        # this is a tad dirty, but it works, and solves the synchronisation problem
        # so keep it. It's not a beauty contest
        # DO convert to ASP though, for convenience
        self.relevant_positions[0][0] = our_logging.RIGHT_ONLY_HUMAN[act] + "."
        #self.relevant_positions.appendleft(our_logging.RIGHT_ONLY_HUMAN[act])

        return act

