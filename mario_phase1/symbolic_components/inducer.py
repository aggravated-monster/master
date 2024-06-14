import subprocess

import psutil

from mario_phase1.mario_logging.logging import Logging


def extract_result(temp_file):
    result = []
    with open(temp_file) as infile:
        content = infile.readlines()
        for info in content:
            if info.startswith('\n') or info.startswith('%%%') or info.startswith('Pre-processing'):
                break
            result.append(info.strip('\n'))
    return result


class Inducer:

    def __init__(self, config, prior_knowledge=None):
        super().__init__()

        self.bias = config["bias"]
        self.forget = config["forget"]
        self.constraints = config["constraints"]
        self.knowledge_conjunction_dict = dict()

        self.ilasp_binary = config["ilasp_binary"]
        self.ilasp_mode_bias = []
        with open(config["ilasp_mode_bias"]) as f:
            for line in f:
                self.ilasp_mode_bias.append(line.strip())

        self.induced_asp_logger = Logging.get_logger('induced_asp')
        self.ilasp_program_logger = Logging.get_logger('ilasp_program')
        self.partial_interpretations_neg_logger = Logging.get_logger('partial_interpretations_neg')
        self.partial_interpretations_pos_logger = Logging.get_logger('partial_interpretations_pos')
        if prior_knowledge is None:
            self.knowledge = []
        else:
            self.knowledge = prior_knowledge

        self.__load_previous_knowledge()

    def learn(self):
        # load previously learned program
        self.__load_previous_knowledge()
        # get the examples
        positives, negatives = self.__get_examples()
        # clean up inconsistencies
        positives_clean, negatives_clean = self.__remove_inconsistencies(positives, negatives)
        # merge the lists together with the background and searchspace
        program_file_name = self.__write_ilasp_program(positives_clean, negatives_clean)
        # try the induction
        induced_knowledge = self.__induce(program_file_name)
        # consolidate the induced knowledge with the knowledge already acquired
        if self.constraints:
            self.__consolidate_constraints(induced_knowledge)
        else:
            self.__consolidate_positive_rules(induced_knowledge)
        return self.knowledge


    def __get_examples(self):
        # get the file handlers of the positive and negative example loggers
        rfh_partial_interpretations_pos = self.partial_interpretations_pos_logger.handlers[0]
        rfh_partial_interpretations_neg = self.partial_interpretations_neg_logger.handlers[0]
        # dirty read the log file
        positives = []
        negatives = []
        with open(rfh_partial_interpretations_pos.baseFilename) as f:
            for line in f:
                positives.append(line.strip())
        with open(rfh_partial_interpretations_neg.baseFilename) as f:
            for line in f:
                negatives.append(line.strip())

        return positives, negatives

    def __consolidate_positive_rules(self, induced_knowledge):
        # add the induced knowledge to the knowledge already acquired.
        # We do this rather naively, following these rules:
        # - rules with only a head are skipped. The mode bias allows for actions in the heads only anyway, and the
        # actions are already known, so no need to make them more explicit than they already are
        # - constraints are allowed only if they contain at least 2 atoms
        print("Old knowledge: " + str(self.knowledge))
        print("Induced knowledge: " + str(induced_knowledge))

        #self.knowledge.extend(x for x in induced_knowledge if x not in self.knowledge and len(x.split()) > 2) # 3 words allows for head :- body, body, as well as :- body, body
        if self.forget:
            # only keep the new knowledge and forget the old
            self.knowledge = [x for x in induced_knowledge if
                              len(x.split()) > 2]  # 3 words allows for head :- body, body, as well as :- body, body
        else:
            raise NotImplementedError("Memory retention not implemented")


        print("New knowledge: " + str(self.knowledge))

    def __consolidate_constraints(self, induced_knowledge):
        # add the induced knowledge to the knowledge already acquired.
        # We do this rather naively, following these rules:
        # - rules with only a head are skipped. The mode bias doesn't allow for actions in the head so
        # this should not happen in the first place
        # - constraints are allowed only if they contain at least 2 atoms
        print("Old knowledge: " + str(self.knowledge))
        print("Induced knowledge: " + str(induced_knowledge))

        if self.forget:
            # only keep the new knowledge and forget the old
            self.knowledge = [x for x in induced_knowledge if len(x.split()) > 2]
        else:
            raise NotImplementedError("Memory retention not implemented")


        print("New knowledge: " + str(self.knowledge))


    def __remove_inconsistencies(self, positives, negatives):

        if self.bias == 'positive':
            # remove from negatives only
            return positives, [x for x in negatives if x.replace("neg", "pos") not in positives]

        if self.bias == 'negative':
            # remove from positives only
            return [x for x in positives if x.replace("pos", "neg") not in negatives], negatives

        # default: remove from both
        return ([x for x in positives if x.replace("pos", "neg") not in negatives],
                [x for x in negatives if x.replace("neg", "pos") not in positives])

    def __write_ilasp_program(self, positives, negatives):

        # roll over. This ensures we keep all previous attempts, even if there is no hope for learning anything
        frh_ilasp_program = self.ilasp_program_logger.handlers[0]
        frh_ilasp_program.doRollover()

        ilasp_program = self.ilasp_mode_bias + positives + negatives
        # offload to fresh file
        for line in ilasp_program:
            self.ilasp_program_logger.info(line)

        return frh_ilasp_program.baseFilename

    def __induce(self, ilasp_program):
        temp_file = "result.tmp"
        execution_string = self.ilasp_binary + "  --version=4  " + ilasp_program + " > " + temp_file
        ilasp_process = subprocess.Popen(execution_string, shell=True)
        p = psutil.Process(ilasp_process.pid)
        try:
            p.wait(timeout=600)
        except psutil.TimeoutExpired:
            p.kill()
            print("Learner timeout. Process killed.")
            return None

        result = extract_result(temp_file)

        return result

    def __load_previous_knowledge(self):

        # temporarily forget everything to avoid  dups
        self.knowledge = []
        # load the current asp program. This is what we know so far.
        rfh_induced_asp = self.induced_asp_logger.handlers[0]
        with open(rfh_induced_asp.baseFilename) as f:
            for line in f:
                self.knowledge.append(line.strip())
