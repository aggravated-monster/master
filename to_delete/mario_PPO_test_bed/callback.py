import json
import os
from stable_baselines3_master.stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, config, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        # keep track of the configuration used for a training session
        self.config = config

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            with open(self.save_path+"/config.json", "w") as outfile:
                json.dump(self.config, outfile, indent=4)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
