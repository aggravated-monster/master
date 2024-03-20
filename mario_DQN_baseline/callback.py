import json
import os
from stable_baselines3_master.stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3_master.stable_baselines3.common.logger import TensorBoardOutputFormat

from mario_DQN_baseline.our_logging.our_logging import Logging


class CheckpointCallback(BaseCallback):

    def __init__(self, check_freq, save_path, config, model_name, verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_name = model_name
        # keep track of the configuration used for a training session
        self.config = config

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            with open(self.save_path + "/configuration.json", "w") as outfile:
                json.dump(self.config, outfile, indent=4)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, self.model_name, '_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


# callback die zorgt voor verbinding en grafiek in tensorboard
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


# per episode
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

                # self.tb_formatter.writer.add_scalar("distance/episode/env #{}".format(i + 1),
                #                                     distance,
                #                                     self.count)
                # self.tb_formatter.writer.add_scalar("velocity/episode/env #{}".format(i + 1),
                #                                     velocity,
                #                                     self.count)
                # self.tb_formatter.writer.add_scalar("score/episode/env #{}".format(i + 1),
                #                                     score,
                #                                     self.count)
                # self.tb_formatter.writer.add_scalar("reward/episode/env #{}".format(i + 1),
                #                                     reward,
                #                                     self.count)

                self.our_logger.info(self.episode_log_template.format(env=(i + 1),
                                                                      episode=self.count,
                                                                      distance=distance,
                                                                      time=time,
                                                                      score=score,
                                                                      steps=self.num_timesteps,
                                                                      reward=reward,
                                                                      ))

        return True
