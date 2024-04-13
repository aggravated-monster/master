import logging.config
import os
from datetime import datetime

import yaml

RIGHT_ONLY_HUMAN = [
    'noop',
    'right',
    'jump',
    'sprint',
    'long_jump'
]


class Logging(object):

    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d-%H.%M.%S")

    def configure(self):

        with open('./mario_logging/logging_config.yaml', 'rt') as f:
            config = yaml.safe_load(f.read())

        for k, v in config["handlers"].items():
            # search for file handlers having a filename key.
            # The value of this key may contain a placeholder which we replace for the unique run_id
            if 'filename' in v.keys():
                config["handlers"][k]["filename"] = config["handlers"][k]["filename"].format(run_id=str(self.run_id))

        # Now that we have the manipulated configdict, configure the python mario_logging module
        logging.config.dictConfig(config)

    def configure_for_experiments(self, name=''):

        with open('../mario_logging/logging_config_experiments.yaml', 'rt') as f:
            config = yaml.safe_load(f.read())

        for k, v in config["handlers"].items():
            # search for file handlers having a filename key.
            # The value of this key may contain a placeholder which we replace for the unique run_id
            if 'filename' in v.keys():
                config["handlers"][k]["filename"] = config["handlers"][k]["filename"].format(
                    run_id=str(self.run_id) + '_' + name)

        # Now that we have the manipulated configdict, configure the python mario_logging module
        logging.config.dictConfig(config)

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        return logger

    # for those who want to know
    def get_run_id(self):
        return self.run_id


loggingClass = Logging


def initialize(for_experiments: bool = False, name: str = ''):
    timing_dir = "./logs/timing"
    timing_symbolic_dir = "./logs/timing/symbolic"
    explanation_dir = "./logs/explain"
    if for_experiments:
        partial_interpretations_dir = "../asp/examples"
        advice_program_dir = "../asp/advice"
    else:
        partial_interpretations_dir = "./asp/examples"
        advice_program_dir = "./asp/advice"

    if not os.path.exists(timing_dir):
        # Create the directory
        os.makedirs(timing_dir)
        logging.info("Timings directory created")
    else:
        logging.info("Timings directory already exists")

    if not os.path.exists(timing_symbolic_dir):
        # Create the directory
        os.makedirs(timing_symbolic_dir)
        logging.info("Timings symbolic directory created")
    else:
        logging.info("Timings symbolic directory already exists")

    if not os.path.exists(explanation_dir):
        # Create the directory
        os.makedirs(explanation_dir)
        logging.info("Explanation directory created")
    else:
        logging.info("Game directory already exists")

    if not os.path.exists(partial_interpretations_dir):
        # Create the directory
        os.makedirs(partial_interpretations_dir)
        logging.info("Partial interpretations directory created")
    else:
        logging.info("Partial interpretations directory already exists")

    if not os.path.exists(advice_program_dir):
        # Create the directory
        os.makedirs(advice_program_dir)
        logging.info("Advice asp directory created")
    else:
        logging.info("Advice asp directory already exists")

    if for_experiments:
        loggingClass().configure_for_experiments(name=name)
    else:
        loggingClass().configure()
