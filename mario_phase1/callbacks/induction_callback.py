from codetiming import Timer

from mario_phase1.callbacks.callback import BaseCallback
from mario_phase1.mario_logging.logging import Logging


class InductionCallback(BaseCallback):

    def __init__(self, inducer, advisor, check_freq, max_induced_programs):
        super(InductionCallback, self).__init__()
        self.check_freq = check_freq
        self.max_induced_programs = max_induced_programs
        self.inducer = inducer
        self.advisor = advisor
        self.induction_logger = Logging.get_logger('try_induction')
        self.induction_logger_template = "{seed};{attempt},{episode};"
        self.induced_asp_logger = Logging.get_logger('induced_asp')
        self.negative_examples_logger = Logging.get_logger('examples_negative')
        self.positive_examples_logger = Logging.get_logger('examples_positive')
        self.attempt = 0
        self.successes = 0

    def _on_episode(self) -> bool:
        # perhaps best is to try the induction on episode, as Mario would otherwise weirdly halt in the middle of his actions
        return True

    def _on_step(self) -> bool:
        if (self.n_calls % self.check_freq == 0) and self.successes < self.max_induced_programs:
            self.attempt += 1
            text = self.induction_logger_template.format(seed=self.model.seed,
                                                         attempt=self.attempt,
                                                         episode=self.n_episodes) + '{:0.8f}'

            with Timer(name="Induction timer", text=text, logger=self.induction_logger.info):
                result = self.inducer.learn()

                # try out if an open log file can be written to
                result.append("long_jump :- close.")

                # Mario learned something if there is a result
                # For now, we will then roll over everything, and start fresh.
                # This makes things easier to track
                if len(result) > 0:
                    # if result has yielded anything, write to clingo file
                    # TODO determine if we need to rollover the examples as well
                    rfh_induced_asp = self.induced_asp_logger.handlers[0]
                    rfh_positive_examples = self.positive_examples_logger.handlers[0]
                    rfh_negative_examples = self.negative_examples_logger.handlers[0]
                    rfh_induced_asp.doRollover()
                    rfh_positive_examples.doRollover()
                    rfh_negative_examples.doRollover()
                    for line in result:
                        self.induced_asp_logger.info(line)
                    self.advisor.refresh()
                    self.successes += 1

        return True
