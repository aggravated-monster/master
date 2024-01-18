import pandas as pd
from ultralytics import YOLO

class Detector:

    def __init__(self):
        super().__init__()

    def detect(self, observation) -> pd.DataFrame:
        # YOLO detection
        # TODO change model to final one
        model = YOLO('../mario_phase0/models/Mario_OD_vanilla_best.pt')

        results = model(observation)

        # what if there are no detections?
        positions = pd.DataFrame(data=None, columns=['name', 'xmin', 'xmax', 'ymin', 'ymax'])

        for r in results:
        #     TODO convert classes into names
            boxes = r.boxes.numpy()
            classes = pd.DataFrame(boxes.cls, columns=['class'])
            # other types of bounding box data can be chosen: xyxy, xywh, xyxyn, xywhn
            xywh = pd.DataFrame(boxes.xywh, columns=['x', 'y', 'w', 'h'])
            positions = pd.concat([classes, xywh], axis=1)

        return positions
