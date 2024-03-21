import clingo
import pandas as pd
from pandas import DataFrame
import warnings

class Positioner:

    def __init__(self, config):
        super().__init__()

        f = open(config["positions_asp"])
        self.positions = f.read()
        f.close()

        f = open(config["show_asp"])
        self.show = f.read()
        f.close()

    def position(self, locations: DataFrame):
        df = locations[['name', 'xmin', 'xmax', 'ymin', 'ymax']].copy()
        template = "name(xmin,xmax,ymin,ymax)."
        # transform dataframe entries to facts
        df['xmin'] = df.apply(
            lambda row: template.replace('name', row['name']).replace('xmin', str(int(row['xmin']))).replace('xmax',
                                                                                                             str(int(
                                                                                                                 row[
                                                                                                                     'xmax']))).replace(
                'ymin', str(int(row['ymin']))).replace('ymax', str(int(row['ymax']))), axis=1)
        # string them together
        current_facts = ' '.join(df['xmin'].tolist())
        # pass to solver.
        clingo_symbols = Solver().solve(self.positions, self.show, current_facts)

        symbols = list(map(lambda x: self.convert_symbol_to_term(x), clingo_symbols))

        return symbols

    def convert_symbol_to_term(self, symbol: clingo.Symbol):
        name = symbol.name
        arguments = symbol.arguments

        term = "" + name + "("
        argstring = ",".join(map(str, arguments))
        term += argstring
        term += ")."

        return term


class Solver:
    def __init__(self):
        super().__init__()
        self.atoms = []

    def solve(self, positions, show, locations):

        control = clingo.Control(message_limit=0)
        control.configuration.solve.models = 1

        # add asp
        control.add("base", [], positions)
        control.add("base", [], locations)
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
        # print("Found solution:", model)
        symbols = model.symbols(shown=True)
        for symbol in symbols:
            # print(symbol)
            self.atoms.append(symbol)



