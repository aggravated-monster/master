from abc import abstractmethod

from mario_phase1.mario_logging.logging import Logging


class ExampleCollector:

    def __init__(self):
        super().__init__()
        self.partial_interpretations_logger_neg = Logging.get_logger('partial_interpretations_neg')
        self.partial_interpretation_template_neg = "#neg({inc},{excl},{ctx})."
        self.partial_interpretations_logger_pos = Logging.get_logger('partial_interpretations_pos')
        self.partial_interpretation_template_pos = "#pos({inc},{excl},{ctx})."
        self.example_set_neg = set()
        self.example_set_pos = set()

    def flush_negatives(self):
        rfh_partial_interpretations_neg = self.partial_interpretations_logger_neg.handlers[0]
        rfh_partial_interpretations_neg.doRollover()
        for item in self.example_set_neg:
            self.partial_interpretations_logger_neg.info(item)
        self.example_set_neg.clear()

    def flush_positives(self):
        rfh_partial_interpretations_pos = self.partial_interpretations_logger_pos.handlers[0]
        rfh_partial_interpretations_pos.doRollover()
        for item in self.example_set_pos:
            self.partial_interpretations_logger_pos.info(item)
        self.example_set_pos.clear()

    def offload_negatives(self):
        rfh_partial_interpretations_neg = self.partial_interpretations_logger_neg.handlers[0]
        rfh_partial_interpretations_neg.doRollover()
        for item in self.example_set_neg:
            self.partial_interpretations_logger_neg.info(item)

    def offload_positives(self):
        rfh_partial_interpretations_pos = self.partial_interpretations_logger_pos.handlers[0]
        rfh_partial_interpretations_pos.doRollover()
        for item in self.example_set_pos:
            self.partial_interpretations_logger_pos.info(item)

    @abstractmethod
    def collect_negative_example(self, last_action, last_observation):
        pass

    @abstractmethod
    def collect_positive_example(self, last_action, last_observation):
        pass

    def _extract_context_naive(self, last_observation):
        # the last observation is a string with atoms. These can be further simplified by only taking the
        # predicates and convert them to 0-arity atoms.
        # quick and dirty hard coded stuff: we are actually only interested in things that
        # are close, adjacent or far,
        # which greatly reduces the size of the set.
        ctx = ""
        if "close" in last_observation:
            ctx = ctx.join("close. ")
        if "far" in last_observation:
            ctx = ctx.join("far. ")
        if "adjacent" in last_observation:
            ctx = ctx.join("adjacent. ")

        return ctx

    def _extract_context(self, last_observation):
        # the last observation is a string with atoms. These can be further simplified by only taking the
        # predicates and convert them to 0-arity atoms.
        # quick and dirty hard coded stuff: we are actually only interested in things that
        # are close, adjacent or far,
        # which greatly reduces the size of the set.
        ctx = ""
        if "imminent_death" in last_observation:
            ctx = ctx.join("imminent_death. ")
        if "danger" in last_observation:
            ctx = ctx.join("danger. ")
        if "far" in last_observation:
            ctx = ctx.join("far. ")

        return ctx


class NaiveExampleCollector(ExampleCollector):

    def __init__(self):
        super().__init__()

    def collect_negative_example(self, last_action, last_observation):
        # add the relevant atoms to the example_set.
        ctx = self._extract_context(last_observation)
        if len(ctx) > 0:  # we have a good example
            example = self.partial_interpretation_template_neg.format(inc="{" + last_action + "}",
                                                                      excl="{}",
                                                                      ctx="{" + ctx + "}")
            self.example_set_neg.update([example])

    def collect_positive_example(self, last_action, last_observation):
        # add the relevant atoms to the example_set.
        ctx = self._extract_context(last_observation)
        if len(ctx) > 0:  # we have a good example
            example = self.partial_interpretation_template_pos.format(inc="{" + last_action + "}",
                                                                      excl="{}",
                                                                      ctx="{" + ctx + "}")
            self.example_set_pos.update([example])


class ConstraintsExampleCollector(ExampleCollector):

    def __init__(self):
        super().__init__()

    def collect_negative_example(self, last_action, last_observation):
        # add the relevant atoms to the example_set.
        ctx = self._extract_context(last_observation)
        if len(ctx) > 0:  # we have a good example
            example = self.partial_interpretation_template_neg.format(inc="{}",
                                                                      excl="{}",
                                                                      ctx="{" + last_action + ". " + ctx + "}")
            self.example_set_neg.update([example])

    def collect_positive_example(self, last_action, last_observation):
        # add the relevant atoms to the example_set.
        ctx = self._extract_context(last_observation)
        if len(ctx) > 0:  # we have a good example
            example = self.partial_interpretation_template_pos.format(inc="{}",
                                                                      excl="{}",
                                                                      ctx="{" + last_action + ". " + ctx + "}")
            self.example_set_pos.update([example])
