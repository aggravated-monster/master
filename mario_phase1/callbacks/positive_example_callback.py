
from master_stable_baselines3.common.callbacks import BaseCallback

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class PositiveExampleCallback(BaseCallback):

    def __init__(self, check_freq, offload_freq):
        super(PositiveExampleCallback, self).__init__()
        self.check_freq = check_freq
        self.offload_freq = offload_freq
        self.our_logger = Logging.get_logger('examples_positive')
        self.partial_interpretations_logger = Logging.get_logger('partial_interpretations_pos')
        self.example_log_template = "{timestep};{mario_time};{action};{state}"
        self.partial_interpretation_template = "#pos({inc},{excl},{ctx})."
        self.example_set = set()

    def _on_episode(self) -> bool:
        return True

    def _on_step(self) -> bool:

        if self.n_calls % self.offload_freq == 0:  # frequency to offload the example .las
            for item in self.example_set:
                self.partial_interpretations_logger.info(item)
            self.example_set.clear()

        if self.n_calls % self.check_freq == 0:

            # make sure we are still alive
            if not self.locals['dones'][-1]:

                last_action, last_observation = self.__obtain_relevant_observation()

                if not self.__is_candidate_example:
                    return True

                # From here on, we know the last action taken has progressed Mario in the game
                # without dying, which is a Good Thing
                mario_time = self.locals['infos']['time']

                self.__log_example(self.n_calls, mario_time, last_action, last_observation)

                # add the relevant atoms to the example_set.
                # the last observation is a string with atoms. These can be further simplified by only taking the
                # predicates and convert them to 0-arity atoms.
                # quick and dirty hard coded stuff: we are actually only interested in things that
                # are close, adjacent or far,
                # which greatly reduces the size of the set.
                ctx = ""
                if "close" in last_observation:
                    ctx = ctx.join("close. ")
                if "far" in last_observation:
                    ctx = ctx.join("far. ")
                if "adjacent" in last_observation:
                    ctx = ctx.join("adjacent. ")
                if len(ctx) > 0:  # we have a good example
                    example = self.partial_interpretation_template.format(inc="{"+last_action+"}",
                                                                          excl="{}",
                                                                          ctx="{"+ctx+"}")
                    self.example_set.update([example])

        return True

    def __obtain_relevant_observation(self):
        observations = self.training_env.venv.envs[0].gym_env.env.relevant_positions
        # the queue length is 5, with the last item being the oldest one
        # the zero-th item in queue is the last one added to the queue and therefore the current state
        # But we want the state prior to that, because that was the state in which the last action
        # was chosen, and that action was apparently safe. So, pick index 1
        # convert to ASP format
        last_observation = " ".join(observations[1][1])
        # convert to ASP format
        last_action = observations[1][0]

        return last_action, last_observation

    def __log_example(self, timestep, mario_time, action, observation):
        self.our_logger.info(self.example_log_template.format(timestep=timestep,
                                                              mario_time=mario_time,
                                                              action=action,
                                                              state=observation
                                                              ))

    def __is_candidate_example(self, i):
        # exclude attempts to run into a pipe. This cannot be a positive example
        # or else it will contradict with the negative examples
        # Detect this by not being impressed with low rewards
        reward = self.locals['reward']
        # restrict positive examples to floor level, otherwise we can get false positives while flying
        env_y_pos = self.locals['info']['y_pos']
        return reward >= 10 & env_y_pos < 80
