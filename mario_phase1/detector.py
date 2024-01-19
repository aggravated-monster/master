import pandas as pd
from typing import List
from ultralytics import YOLO
import yaml

class Detector:

    def __init__(self):
        super().__init__()
        self.model = YOLO('models/Mario_OD_vanilla_best.pt')

        # TODO location of final YAML file
        with open('models/data.yaml','r') as file:
            data = yaml.safe_load(file)
            self.names = data['names']

    def detect(self, observation) -> pd.DataFrame:

        results = self.model(observation)
        # what if there are no detections?
        positions = pd.DataFrame(data=None, columns=['name', 'xmin', 'xmax', 'ymin', 'ymax'])

        for r in results:
            boxes = r.boxes.cpu().numpy()
            classes = pd.DataFrame(boxes.cls, columns=['class'])
            # other types of bounding box data can be chosen: xyxy, xywh, xyxyn, xywhn
            xywh = pd.DataFrame(boxes.xywh, columns=['x', 'y', 'w', 'h'])
            names_pd = classes['class'].apply(lambda x: self.names[int(x)])
            positions = pd.concat([classes, names_pd, xywh], axis=1)

        return positions
