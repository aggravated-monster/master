import clingo
from pandas import DataFrame


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

        df = locations[['name', 'xmin']].copy()
        template = "xmin(name,xpos)."
        # transform dataframe entries to facts
        df['xmin'] = df.apply(lambda row: template.replace('name', row['name']).replace('xpos', str(int(row['xmin']))), axis=1)
        # string them together
        current_facts = ' '.join(df['xmin'].tolist())
        # pass to solver.
        symbols = Solver().solve(self.positions, self.show, current_facts)

        return symbols


class Solver:
    def __init__(self):
        super().__init__()
        self.terms = []

    def solve(self, positions, show, locations):

        control = clingo.Control()
        control.configuration.solve.models = 1

        # add asp
        control.add("base", [], positions)
        control.add("base", [], locations)
        control.add("base", [], show)

        control.ground([("base", [])])
        handle = control.solve(on_model=self.on_model)

        if handle.satisfiable:
            return self.terms

        return None

    def on_model(self, model):
        """
        This is the observer callback for the clingo Control object after solving. It is done in a separate thread,
        which is why we use a new instantiation of the Solver class, otherwise its state is not thread safe
        :param model:
        """
        print("Found solution:", model)
        symbols = model.symbols(terms=True)
        for symbol in symbols:
            print(symbol)
            self.terms.append(symbol)
