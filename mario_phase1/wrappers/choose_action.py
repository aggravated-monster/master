from abc import ABC

import numpy as np
from codetiming import Timer
from gym import ActionWrapper

from mario_stable_baselines_phase1.our_logging import our_logging
from mario_stable_baselines_phase1.our_logging.our_logging import Logging


class ChooseAction(ActionWrapper, ABC):
    logger = Logging.get_logger('actions')

    def __init__(self, env, advisor, seed):
        super().__init__(env)
        self.count = 0
        self.advisor = advisor
        # second logger for advice
        self.advice_logger = Logging.get_logger('advice')
        self.advice_log_template = "{timestep},{action};{advice};{action_chosen};{state}"
        self.seed = seed

    #@Timer(name="ChooseAction wrapper timer", text="{:0.8f}", logger=logger.info)
    def action(self, act):

        text = str(self.seed) + ";{:0.8f}"

        with Timer(name="ChooseAction wrapper timer", text=text, logger=self.logger.info):

            self.count += 1
            # retrieve current observation and pass to Advisor, together with the action
            current_facts = " ".join(self.relevant_positions[0][1])
            current_action = our_logging.RIGHT_ONLY_HUMAN[act] + "."
            advice = self.advisor.advise(current_facts, current_action)

            if advice is None:
                advice = "no model"
                # if Advisor returns None, no model was found
                # Proceed with caution.
                # For the moment, we know for a fact that a constraint was broken, and therefore
                # the current action is bad. Pick another one.
                #actions_to_choose = [*range(0, len(our_logging.RIGHT_ONLY_HUMAN), 1)]
                #actions_to_choose.pop(act)
                #action_chosen = rnd.choice(actions_to_choose)
                action_chosen = act

            elif len(advice) == 0:
                # no advice found. Proceed with chosen action
                action_chosen = act

            else:
                # advice found
                # given that this might be a list, choose one
                if np.random.random() > 0.1:
                    action_chosen = np.random.choice(advice)
                    # convert to correct index
                    action_chosen = our_logging.RIGHT_ONLY_HUMAN.index(action_chosen)
                else:
                    action_chosen = act


                advice = " ".join(advice)
            # log the things

            self.__log_advice(str(self.count), current_action, advice, our_logging.RIGHT_ONLY_HUMAN[action_chosen], current_facts)


            # keep track of the action taken, together with the current observation,
            # as they are used to create examples.
            # this is a tad dirty, but it works, and solves the synchronisation problem
            # so keep it. It's not a beauty contest
            # DO convert to ASP though, for convenience
            self.relevant_positions[0][0] = our_logging.RIGHT_ONLY_HUMAN[action_chosen] + "."
            #self.relevant_positions.appendleft(our_logging.RIGHT_ONLY_HUMAN[act])

            return action_chosen

    def __log_advice(self, timestep, action, advice, action_chosen, observation):
        self.advice_logger.info(self.advice_log_template.format(timestep=timestep,
                                                                action=action,
                                                                advice=advice,
                                                                action_chosen=action_chosen,
                                                                state=observation
                                                                ))
