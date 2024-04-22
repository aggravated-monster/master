import logging

import clingo

from mario_phase1.mario_logging.logging import Logging


class Advisor:

    def __init__(self, config):
        super().__init__()

        #self.advice_asp = config["advice_asp"]
        self.induced_asp_logger = Logging.get_logger('induced_asp')

        self.advice = self.__load_advice()

        f = open(config["show_advice_asp"])
        self.show = f.read()
        f.close()

    def refresh(self):
        self.advice = self.__load_advice()

    def advise(self, current_facts: str, action=None):

        # run with constraints
        #clingo_symbols = Solver().solve(self.advice, self.show, current_facts, action)
        # run without constraimts
        clingo_symbols = Solver().solve(self.advice, self.show, current_facts)

        if clingo_symbols is None:
            # no model found. In case of the ideal advice, this means a constraint is broken
            return None

        symbols = list(map(lambda x: self.convert_symbol_to_term(x), clingo_symbols))

        return symbols

    def convert_symbol_to_term(self, symbol: clingo.Symbol):
        # we are only interested in the 0-arity actions, so no arguments needed
        return symbol.name

    def __load_advice(self):
        advice = []
        rfh_induced_asp = self.induced_asp_logger.handlers[0]
        induced_asp_filename = rfh_induced_asp.baseFilename
        with open(induced_asp_filename) as f:
            for line in f:
                advice.append(line.strip())
            return ' '.join(advice)


class Solver:
    def __init__(self):
        super().__init__()
        self.atoms = []

    def solve(self, advice, show, facts, action=None):

        control = clingo.Control(message_limit=0)
        control.configuration.solve.models = 5

        # add asp
        control.add("base", [], advice)
        control.add("base", [], facts)
        if action is not None:
            control.add("base", [], action)
        control.add("base", [], show)

        control.ground([("base", [])])

        handle = control.solve(on_model=self.on_model)

        if handle.satisfiable:
            return self.atoms

        return None

    def on_model(self, model):
        """
        This is the observer callback for the clingo Control object after solving. It is done in a separate thread,
        which is why we use a new instantiation of the Solver class, otherwise its state is not thread safe
        :param model:
        """
        symbols = model.symbols(shown=True)
        for symbol in symbols:
            # print(symbol)
            self.atoms.append(symbol)