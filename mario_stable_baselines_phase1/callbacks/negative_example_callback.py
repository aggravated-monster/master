import json
import os
from master_stable_baselines3.common.callbacks import BaseCallback
from master_stable_baselines3.common.logger import TensorBoardOutputFormat

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class NegativeExampleCallback(BaseCallback):

    def __init__(self):
        super(NegativeExampleCallback, self).__init__()
        self.count = 0
        self.our_logger = Logging.get_logger('examples_negative')
        self.example_log_template = "{env};{timestep};{episode};{mario_time};{action};{state}"

    def _on_step(self) -> bool:
        # end of episode can mean 2 things: win, or more likely, death
        if self.locals['dones'][-1]:
            self.count += 1

            # presumably 'done' holds the game won flag (pun intended)
            # we only do stuff when Mario did not win the game
            if not self.locals['done']:

                last_action, last_observation = self.__obtain_relevant_observation()
                # preluding on skipping the plunges into the holes
                if last_action is not None:

                    for i in range(self.locals['env'].num_envs):

                        # So, Mario died. There are 2 cases now:
                        # 1. he ran out of time. This is a tricky one. Clearly, running out of time while moving forward in
                        # an open field is not a negative example, whereas repeatedly bumping into a pipe is
                        # Need to think about this one
                        # 2. he ran into an enemy or a hole.
                        # This is clearly a negative example
                        # Both cases can be distinguished by the Mario clock
                        mario_time = self.locals['infos'][i]['time']

                        if mario_time > 0:
                            self.__log_example(i + 1, self.n_calls, self.count, mario_time, last_action,
                                               last_observation)

        return True

    def __obtain_relevant_observation(self):
        observations = self.training_env.venv.envs[0].gym_env.env.relevant_positions

        for observation in observations:
            print(observation)

        # the zero-th item in queue is last one added to the queue, but this is already the new game
        # the first item in the queue was therefore the last state of the lost game.
        # the second item is the state in which Mario chose the action that led to his demise
        # unless Mario is not in the list
        # This happens with holes, where mario disappears already in the last state of the game,
        # so for those we need to look back further, even though this is actually more than 4 frames in the past
        last_observation = observations[2][1]
        last_action = observations[2][0]
        has_mario = any("mario" in s for s in last_observation)
        if not has_mario:
            # there is a case for simply returning here. If Mario learns to jump goombas, he knows
            # to jump holes (and pipes for that matter), as they are all obstacles
            # the risk of trying to catch the hole deaths is that the chosen action while in the hole
            # was to jump, creating a contradiction.
            # and picking the observation just before the hole may be too far, as we skip 4 frames
            # choiceschoices
            # return None, None
            last_observation = observations[4][1]
            last_action = observations[4][0]

        # convert to ASP format
        last_observation = " ".join(last_observation)

        return last_action, last_observation

    def __log_example(self, env, timestep, episode, mario_time, action, observation):
        self.our_logger.info(self.example_log_template.format(env=env,
                                                              timestep=timestep,
                                                              episode=episode,
                                                              mario_time=mario_time,
                                                              action=action,
                                                              state=observation
                                                              ))
