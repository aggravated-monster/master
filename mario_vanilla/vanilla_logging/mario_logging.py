import logging.config
import os
from datetime import datetime
from typing import Dict, Tuple, Any

import numpy as np
import torch
import yaml

RIGHT_ONLY_HUMAN = [
    'noop',
    'right',
    'jump',
    'sprint',
    'long_jump'
]


class Logging(object):

    def __init__(self, with_timing):
        self.run_id = datetime.now().strftime("%Y%m%d-%H.%M.%S")
        self.with_timing = with_timing

    def configure(self, name="train"):

        if self.with_timing:
            with open('./vanilla_logging/logging_config_timing.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())
        else:
            with open('./vanilla_logging/logging_config.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())

        for k, v in config["handlers"].items():
            # search for file handlers having a filename key.
            # The value of this key may contain a placeholder which we replace for the unique run_id
            if 'filename' in v.keys():
                config["handlers"][k]["filename"] = config["handlers"][k]["filename"].format(run_id=str(self.run_id) + '_' + name)

        # Now that we have the manipulated configdict, configure the python vanilla_logging module
        logging.config.dictConfig(config)

    def configure_for_experiments(self, name=''):

        if self.with_timing:
            with open('/logging_config_timing.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())
        else:
            with open('/logging_config.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())

        for k, v in config["handlers"].items():
            # search for file handlers having a filename key.
            # The value of this key may contain a placeholder which we replace for the unique run_id
            if 'filename' in v.keys():
                config["handlers"][k]["filename"] = config["handlers"][k]["filename"].format(
                    run_id=str(self.run_id) + '_' + name)

        # Now that we have the manipulated configdict, configure the python vanilla_logging module
        logging.config.dictConfig(config)

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        return logger

    # for those who want to know
    def get_run_id(self):
        return self.run_id


loggingClass = Logging


def initialize(with_timing: bool = False, for_purpose: str = "TRAIN", name: str = ''):
    timing_dir = "./logs/timing"

    if with_timing:
        if not os.path.exists(timing_dir):
            # Create the directory
            os.makedirs(timing_dir)
            logging.info("Timings directory created")
        else:
            logging.info("Timings directory already exists")

    if for_purpose == "EXPERIMENT":
        loggingClass(with_timing).configure_for_experiments(name=name)
    if for_purpose == "RUN":
        loggingClass(with_timing).configure(name="run")
    else:
        loggingClass(with_timing).configure()