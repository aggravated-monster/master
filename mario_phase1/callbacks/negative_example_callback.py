
from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging


class NegativeExampleCallback(BaseCallback):

    def __init__(self, collector, offload_freq=100):
        super(NegativeExampleCallback, self).__init__()
        self.collector = collector
        self.offload_freq = offload_freq
        self.example_logger = Logging.get_logger('examples_negative')
        self.example_log_template = "{seed};{total_steps};{episode};{episode_steps};{mario_time};{action};{state}"

    def _on_step(self) -> bool:
        # to keep sort of a heartbeat with the positive examples,
        # offload in steps frequency, instead of episodes
        if self.n_calls % self.offload_freq == 0:  # frequency to offload the example .las
            self.collector.flush_negatives()
        return True

    def _on_episode(self) -> bool:

        # end of episode can mean 2 things: win, or more likely, death
        # we only do stuff when Mario did not win the game
        if not self.locals['info']['flag_get']:

            last_action, last_observation, last_env_xpos = self.__obtain_relevant_observation()
            # preluding on skipping the plunges into the holesexample_set
            if (last_action is not None) and self.__is_candidate_example(last_env_xpos):

                # So, Mario died. There are 2 cases now:
                # 1. he ran out of time. This is a tricky one. Clearly, running out of time while moving forward in
                # an open field is not a negative example, whereas repeatedly bumping into a pipe is
                # Need to think about this one
                # 2. he ran into an enemy or a hole.
                # This is clearly a negative example
                # Both cases can be distinguished by the Mario clock
                mario_time = self.locals['info']['time']

                # ran into enemy or hole
                if mario_time > 0:
                    self.__log_example(self.model.seed, self.num_timesteps_done, self.n_episodes, self.locals['episode_step_counter'],
                                       mario_time,
                                       last_action,
                                       last_observation)

                    self.collector.collect_negative_example(last_action, last_observation)

        return True

    def __obtain_relevant_observation(self):
        observations = self.training_env.env.env.env.relevant_positions
        # the zero-th item in queue is last one added to the queue, was therefore the last state of the lost game.
        # the first item is the state in which Mario chose the action that led to his demise
        # unless Mario is not in the list
        # This happens with holes, where mario disappears already in the last state of the game,
        # so for those we need to look back further, even though this is actually more than 4 frames in the past
        last_observation = observations[1][1]
        last_action = observations[1][0]
        last_env_xpos = observations[1][2]
        has_mario = any("mario" in s for s in last_observation)
        if not has_mario:
            # there is a case for simply returning here. If Mario learns to jump goombas, he knows
            # to jump holes (and pipes for that matter), as they are all obstacles
            # the risk of trying to catch the hole deaths is that the chosen action while in the hole
            # was to jump, creating a contradiction.
            # and picking the observation just before the hole may be too far, as we skip 4 frames
            # Also, closer inspection has learned that holes are very confidently jumped into, quite
            # often, as it really requires to calculate trajectory in order to say anything useful
            # about Mario's choices. This will create massive contradictions.
            # So, best to indeed ignore them while not using trajectories.
            return None, None, last_env_xpos
            #last_observation = observations[4][1]
            #last_action = observations[4][0]

        # convert to ASP format
        last_observation = " ".join(last_observation)

        return last_action, last_observation, last_env_xpos

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
        # exclude examples where Mario accidentally landed just before an enemy after a jump
        # This cannot be marked as a negative, as it is a timing issue which we don´t do in this thesis
        # restrict negative examples to floor level, otherwise we can get false negatives while flying
        env_x_pos = self.locals['info']['x_pos']
        env_y_pos = self.locals['info']['y_pos']
        return (env_x_pos - last_env_xpos < 60) and (env_y_pos < 80)
