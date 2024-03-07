import logging.config
import uuid
import yaml


class Logging(object):

    def __init__(self, with_timing):
        self.run_id = uuid.uuid1()
        self.with_timing = with_timing

    def configure(self):

        if self.with_timing:
            with open('logging_config_timing.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())
        else:
            with open('logging_config.yaml', 'rt') as f:
                config = yaml.safe_load(f.read())

        for k, v in config["handlers"].items():
            # search for file handlers having a filename key.
            # The value of this key may contain a placeholder which we replace for the unique run_id
            if 'filename' in v.keys():
                config["handlers"][k]["filename"] = config["handlers"][k]["filename"].format(run_id=str(self.run_id))

        # Now that we have the manipulated configdict, configure the python logging module
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
    loggingClass(with_timing).configure()
