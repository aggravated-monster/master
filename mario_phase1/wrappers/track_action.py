from abc import ABC

from gym import ActionWrapper

from mario_stable_baselines_phase1.our_logging import our_logging


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

        # We are not interested in the positions and actions chosen while Mario is in a state
        # where choosing actions does not change his course.
        # This happens in-flight.
        # To make things easier, we will only record actions and positions that take place at ground-level
        # This should be sufficient to encounter all different scenarios that can be generalised over
        # not-ground-level positions
        # Mario's y-pos can be obtained from the SuperMarioBrosEnv, in protected property _y_position
        # This is terribly ugly, but given time constraints, we cannot redesign the environment
        if self.unwrapped.env._y_position < 80:
            self.relevant_positions[0][0] = our_logging.RIGHT_ONLY_HUMAN[act]
            #self.relevant_positions.appendleft(our_logging.RIGHT_ONLY_HUMAN[act])

        return act

