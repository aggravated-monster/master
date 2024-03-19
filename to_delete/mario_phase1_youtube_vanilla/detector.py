import pandas as pd
from typing import List
from ultralytics import YOLO
import yaml

class Detector:

    def __init__(self):
        super().__init__()
        self.model = YOLO('../../Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt')

        # TODO location of final YAML file
        with open('../../Object_detector/models/data.yaml', 'r') as file:
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
            classes['name'] = classes['class'].apply(lambda x: self.names[int(x)])
            positions = pd.concat([classes, xywh], axis=1)

        return positions
