import clingo
import pandas as pd
from pandas import DataFrame

from mario_phase1.mario_logging.logging import Logging


class Positioner:

    def __init__(self, config):
        super().__init__()

        self.generate_examples = config["generate_examples"]

        f = open(config["positions_asp"])
        self.positions = f.read()
        f.close()

        f = open(config["show_asp"])
        self.show = f.read()
        f.close()

        if self.generate_examples:
            f = open(config["relative_positions_asp"])
            self.relative_positions = f.read()
            f.close()

            f = open(config["show_closest_obstacle_asp"])
            self.show_closest_obstacle = f.read()
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
        if self.relative_positions:
            clingo_symbols = Solver().solve(self.positions, self.show, current_facts, self.relative_positions, self.show_closest_obstacle)
        else:
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

        # correct 0-arity predicates
        term = term.replace('()', "")

        return term


class Solver:
    def __init__(self):
        super().__init__()
        self.atoms = []

    def solve(self, positions, show, locations, relative_positions=None, show_closest_obstacles=None):

        control = clingo.Control()
        control.configuration.solve.models = 1

        # add asp
        control.add("base", [], positions)
        control.add("base", [], locations)
        control.add("base", [], show)
        if relative_positions is not None:
            control.add("relative", [], relative_positions)
            control.add("relative", [], show_closest_obstacles)

        control.ground([("base", [])])

        if relative_positions is not None:
            control.ground([("relative", [])])

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
