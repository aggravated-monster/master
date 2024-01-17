import pandas as pd


class Detector:

    def __init__(self):
        super().__init__()

    def detect(self, observation) -> pd.DataFrame:
        # TODO implement trained detection network
        # for now, stub the response
        # quick & dirty.
        data = [['mario', 10, 15, 80, 90], ['enemy', 15, 30, 80, 90], ['gap', 35, 45, 80, 120]]

        # Create the pandas DataFrame
        positions = pd.DataFrame(data, columns=['name', 'xmin', 'xmax', 'ymin', 'ymax'])

        return positions
