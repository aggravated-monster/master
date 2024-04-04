import regex as re

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging


class PositiveExampleCallback(BaseCallback):

    def __init__(self, check_freq, offload_freq):
        super(PositiveExampleCallback, self).__init__()
        self.check_freq = check_freq
        self.offload_freq = offload_freq
        self.example_logger = Logging.get_logger('examples_positive')
        self.partial_interpretations_logger = Logging.get_logger('partial_interpretations_pos')
        self.example_log_template = "{seed};{total_steps};{episode};{episode_steps};{mario_time};{action};{state}"
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
            if not self.locals['done']:

                last_action, last_observation, last_env_x_pos = self.__obtain_relevant_observation()

                # A small intervention, with the following justification:
                # The reward function calculates the distance traveled from one observation to the other
                # Normally, a NOOP would then yield 0, but as we skip 4 frames, a NOOP in many cases
                # yields more than 0 and even more than 10.
                if last_action == 'noop.':
                    return True

                if not self.__is_candidate_example(last_env_x_pos):
                    return True

                # From here on, we know the last action taken has progressed Mario in the game
                # without dying, which is a Good Thing
                mario_time = self.locals['info']['time']

                self.__log_example(self.model.seed, self.num_timesteps_done, self.n_episodes,
                                   self.locals['episode_step_counter'],
                                   mario_time,
                                   last_action,
                                   last_observation)

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
                    # TODO debatable. The absence of any obstacle could also qualify as a good example perhaps?
                    example = self.partial_interpretation_template.format(inc="{" + last_action + "}",
                                                                          excl="{}",
                                                                          ctx="{" + ctx + "}")
                    self.example_set.update([example])

        return True

    def __obtain_relevant_observation(self):
        observations = self.training_env.env.env.env.relevant_positions
        # the queue length is 5, with the last item being the oldest one
        # the zero-th item in queue is the last one added to the queue and therefore the current state
        # But we want the state prior to that, because that was the state in which the last action
        # was chosen, and that action was apparently safe. So, pick index 1
        # convert to ASP format
        last_observation = " ".join(observations[1][1])
        # convert to ASP format
        last_action = observations[1][0]

        last_env_x_pos = observations[1][2]

        return last_action, last_observation, last_env_x_pos

    def __log_example(self, seed, total_steps, episode, episode_steps, mario_time, action, observation):
        self.example_logger.info(self.example_log_template.format(seed=seed,
                                                                  total_steps=total_steps,
                                                                  episode=episode,
                                                                  episode_steps=episode_steps,
                                                                  mario_time=mario_time,
                                                                  action=action,
                                                                  state=observation
                                                                  ))

    def __is_candidate_example(self, last_env_xpos):

        # exclude attempts to run into a pipe. This cannot be a positive example
        # or else it will contradict with the negative examples
        # Detect this by not being impressed with low progression
        reward = self.locals['reward']
        # restrict positive examples to floor level, otherwise we can get false positives while flying
        env_x_pos = self.locals['info']['x_pos']
        env_y_pos = self.locals['info']['y_pos']
        return (env_x_pos - last_env_xpos > 10) and (env_y_pos < 80)

