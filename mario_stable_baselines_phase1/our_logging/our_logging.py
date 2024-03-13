import logging.config
import os
import uuid
from datetime import datetime

import yaml

RIGHT_ONLY_HUMAN = [
    'noop',
    'right',
    'jump',
    'sprint',
    'long_jump',
]


class Logging(object):

    def __init__(self, with_timing):
        self.run_id = datetime.now().strftime("%Y%m%d-%H.%M.%S")
        self.with_timing = with_timing

    def configure(self):

        if self.with_timing:
            with open('./our_logging/logging_config_timing.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())
        else:
            with open('./our_logging/logging_config.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())

        for k, v in config["handlers"].items():
            # search for file handlers having a filename key.
            # The value of this key may contain a placeholder which we replace for the unique run_id
            if 'filename' in v.keys():
                config["handlers"][k]["filename"] = config["handlers"][k]["filename"].format(run_id=str(self.run_id))

        # Now that we have the manipulated configdict, configure the python our_logging module
        logging.config.dictConfig(config)

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        return logger

    # for those who want to know
    def get_run_id(self):
        return self.run_id


loggingClass = Logging


def initialize(with_timing: bool = False):

    timing_dir = "./logs/timing"
    examples_dir = "./logs/examples"

    if not os.path.exists(timing_dir):
        # Create the directory
        os.makedirs(timing_dir)
        logging.info("Timings directory created")
    else:
        logging.info("Timings directory already exists")

    if not os.path.exists(examples_dir):
        # Create the directory
        os.makedirs(examples_dir)
        logging.info("Examples directory created")
    else:
        logging.info("Examples directory already exists")

    loggingClass(with_timing).configure()
