import logging

import clingo


class Advisor:

    def __init__(self, config):
        super().__init__()

        f = open(config["advice_asp"])
        self.advice = f.read()
        f.close()

        f = open(config["show_advice_asp"])
        self.show = f.read()
        f.close()


    def advise(self, current_facts: str, action: str):

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


class Solver:
    def __init__(self):
        super().__init__()
        self.atoms = []

    def solve(self, advice, show, facts, action=None):

        control = clingo.Control()
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
        print("Found solution:", model)
        symbols = model.symbols(shown=True)
        for symbol in symbols:
            # print(symbol)
            self.atoms.append(symbol)