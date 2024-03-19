from our_stable_baselines3.common.callbacks import BaseCallback
from our_stable_baselines3.common.logger import TensorBoardOutputFormat

from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class EpisodeCallback(BaseCallback):

    def __init__(self):
        super(EpisodeCallback, self).__init__()
        self.count = 0
        self.our_logger = Logging.get_logger('episodes')
        self.console_logger = Logging.get_logger('console')
        self.console_log_template = "Episode {episode} finished at timestep {steps}."
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
                self.console_logger.info(self.console_log_template.format(episode=self.count, steps=self.num_timesteps))

        return True
