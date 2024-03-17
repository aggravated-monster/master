import json
import os
from stable_baselines3_master.stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3_master.stable_baselines3.common.logger import TensorBoardOutputFormat

from mario_DQN_baseline.our_logging.our_logging import Logging


class CheckpointCallback(BaseCallback):

    def __init__(self, check_freq, save_path, config, verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        # keep track of the configuration used for a training session
        self.config = config

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            with open(self.save_path + "/configuration.json", "w") as outfile:
                json.dump(self.config, outfile, indent=4)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


class IntervalCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def __init__(self, check_freq, verbose=1):
        super(IntervalCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            rewards = self.locals['rewards']
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("reward/step/env #{}".format(i + 1),
                                                    rewards[i],
                                                    self.n_calls)

        return True


class EpisodeCallback(BaseCallback):

    def __init__(self):
        super(EpisodeCallback, self).__init__()
        self.count = 0
        self.our_logger = Logging.get_logger('episodes')
        self.episode_log_template = "{env},{episode},{distance},{time},{score},{steps},{reward}"

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if self.locals['dones'][-1]:
            self.count += 1

            for i in range(self.locals['env'].num_envs):
                distance = self.locals['infos'][i]['x_pos']
                time = self.locals['infos'][i]['time']
                score = self.locals['infos'][i]['score']
                reward = self.locals['rewards'][i]

                # Q: begint dat spel altijd met 400 en is het terugtellen monotoon?
                velocity = distance / (400 - time)

                self.tb_formatter.writer.add_scalar("distance/episode/env #{}".format(i + 1),
                                                    distance,
                                                    self.count)
                self.tb_formatter.writer.add_scalar("velocity/episode/env #{}".format(i + 1),
                                                    velocity,
                                                    self.count)
                self.tb_formatter.writer.add_scalar("score/episode/env #{}".format(i + 1),
                                                    score,
                                                    self.count)
                self.tb_formatter.writer.add_scalar("reward/episode/env #{}".format(i + 1),
                                                    reward,
                                                    self.count)

                self.our_logger.info(self.episode_log_template.format(env=(i + 1),
                                                                      episode=self.count,
                                                                      distance=distance,
                                                                      time=time,
                                                                      score=score,
                                                                      steps=self.num_timesteps,
                                                                      reward=reward
                                                                      ))

        return True


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


class PositiveExampleCallback(BaseCallback):

    def __init__(self, check_freq):
        super(PositiveExampleCallback, self).__init__()
        self.check_freq = check_freq
        self.our_logger = Logging.get_logger('examples_positive')
        self.example_log_template = "{env};{timestep};{mario_time};{action};{state}"

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

            # make sure we are still alive
            if not self.locals['dones'][-1]:

                last_action, last_observation = self.__obtain_relevant_observation()

                for i in range(self.locals['env'].num_envs):

                    if not self.__is_candidate_example(i):
                        return True

                    # From here on, we know the last action taken has progressed Mario in the game
                    # without dying, which is a Good Thing
                    mario_time = self.locals['infos'][i]['time']

                    self.__log_example(i + 1, self.n_calls, mario_time, last_action, last_observation)

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

    def __log_example(self, env, timestep, mario_time, action, observation):
        self.our_logger.info(self.example_log_template.format(env=env,
                                                              timestep=timestep,
                                                              mario_time=mario_time,
                                                              action=action,
                                                              state=observation
                                                              ))

    def __is_candidate_example(self, i):
        # exclude attempts to run into a pipe. This cannot be a positive example
        # or else it will contradict with the negative examples
        # Detect this by not being impressed with low rewards
        reward = self.locals['rewards'][i]
        # restrict positive examples to floor level, otherwise we can get false positives while flying
        env_y_pos = self.locals['infos'][i]['y_pos']
        return reward >= 10 & env_y_pos < 80
