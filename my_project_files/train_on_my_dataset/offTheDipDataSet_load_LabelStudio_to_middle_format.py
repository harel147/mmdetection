import copy
import json
import os

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

@DATASETS.register_module()
class OffTheDipDataset(CustomDataset):

    CLASSES = ('person',)
    #CLASSES = ('bird',)
    #CLASSES = ('person', 'bird')

    studio_label_ann = 'my_project_files/data_training/labels/3.json'

    def load_annotations(self, ann_file):
        with open(self.studio_label_ann) as f:
          d = json.load(f)

        results = d["result"]

        data_infos = []
        frames = []

        for i in range(len(self.CLASSES)):
            for res in results:
                seq = res["value"]["sequence"]
                for f in seq:
                    # label studio format are percentages of overall image dimension, need to translate to real dimentions.
                    pixel_x = f["x"] / 100.0 * IMAGE_WIDTH
                    pixel_y = f["y"] / 100.0 * IMAGE_HEIGHT
                    pixel_width = f["width"] / 100.0 * IMAGE_WIDTH
                    pixel_height = f["height"] / 100.0 * IMAGE_HEIGHT

                    if f["frame"] not in frames:
                        frames.append(f["frame"])
                        data_info = dict(filename=f'{f["frame"]}.jpg', width=1280, height=720, ann=dict(bboxes=[], labels=[], bboxes_ignore=[], labels_ignore=[]))
                        bboxes = [pixel_x, pixel_y, (pixel_x+pixel_width), (pixel_y+pixel_height)]
                        data_info["ann"]["bboxes"].append(bboxes)
                        data_info["ann"]["labels"].append(i)
                        data_infos.append(data_info)

                    else:
                        for d in data_infos:
                            if d["filename"] == f'{f["frame"]}.jpg':
                                bboxes = [pixel_x, pixel_y, (pixel_x+pixel_width), (pixel_y+pixel_height)]
                                d["ann"]["bboxes"].append(bboxes)
                                d["ann"]["labels"].append(i)

        for d in data_infos:
            d["ann"]["bboxes"] = np.array(d["ann"]["bboxes"], dtype=np.float32).reshape(-1, 4)
            d["ann"]["labels"] = np.array(d["ann"]["labels"], dtype=np.long)
            d["ann"]["bboxes_ignore"] = np.array(d["ann"]["bboxes_ignore"], dtype=np.float32).reshape(-1, 4)
            d["ann"]["labels_ignore"] = np.array(d["ann"]["labels_ignore"], dtype=np.long)

        return data_infos
