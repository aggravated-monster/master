import glob
import os

from codetiming import Timer

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging



class InductionCallback(BaseCallback):

    def __init__(self, check_freq, examples_dir="./asp/ilasp/"):
        super(InductionCallback, self).__init__()
        self.check_freq = check_freq
        self.induction_logger = Logging.get_logger('induction')
        self.induce_rules_logger = Logging.get_logger('induce_rules')
        self.induction_log_template = "{seed};{total_steps};{episode};{episode_steps};{mario_time};{action};{state}"
        self.induce_rules_template = "{head} :- {body}."
        self.example_set = set()
        self.examples_dir = examples_dir

    def _on_episode(self) -> bool:
        # perhaps best is to try the induction on episode, as Mario would otherwise weirdly halt in the middle of his actions
        return True

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

            text = str(self.model.seed) + ";{:0.8f}"

            with Timer(name="Induction timer", text=text, logger=self.induction_logger.info):

                positive_pattern = os.path.join(self.examples_dir, '20240405-17.57.09_positive.las')
                negative_pattern = os.path.join(self.examples_dir, '20240405-17.57.09_negative.las')

                # collect the examples
                positive_example_files = glob.glob(positive_pattern)
                negative_example_files = glob.glob(negative_pattern)

                with open(positive_example_files[-1], "r") as positive_file:
                    positive_lines = [line.rstrip() for line in positive_file]

                with open(negative_example_files[-1], "r") as negative_file:
                    negative_lines = [line.rstrip() for line in negative_file]

                # clean up inconsistencies
                # Take the smallest list and search for identical examples in the other list.
                # When a hit is found, add both to the to_delete list

                # try the induction

                # if success, force roll-over and write to clingo file


        return True

