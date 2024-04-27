import cv2
import pandas as pd
import yaml
from ultralytics import YOLO

from mario_phase0.utils import get_current_date_time_string


class Detector:

    def __init__(self, config):
        super().__init__()
        self.model = YOLO(config["detector_model_path"])

        with open(config["detector_label_path"], 'r') as file:
            data = yaml.safe_load(file)
            self.names = data['names']

    def detect(self, observation) -> pd.DataFrame:
        # YOLO detection
        results = self.model(observation, verbose=False)

        # what if there are no detections?
        positions = pd.DataFrame(data=None, columns=['name', 'xmin', 'xmax', 'ymin', 'ymax'])

        for r in results:
            boxes = r.boxes.cpu().numpy()
            classes = pd.DataFrame(boxes.cls, columns=['class'])
            # other types of bounding box data can be chosen: xyxy, xywh, xyxyn, xywhn
            xyxy = pd.DataFrame(boxes.xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
            classes['name'] = classes['class'].apply(lambda x: self.names[int(x)])
            positions = pd.concat([classes, xyxy], axis=1)

            #masks = r.masks  # Masks object for segmentation masks outputs
            #keypoints = r.keypoints  # Keypoints object for pose outputs
            #probs = r.probs  # Probs object for classification outputs
            #r.show()  # display to screen
            #r.save(filename='./frames/img_mario1-1_' + get_current_date_time_string() + '.png')  # save to disk

        return positions
