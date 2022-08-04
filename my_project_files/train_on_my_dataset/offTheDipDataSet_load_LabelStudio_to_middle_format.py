import copy
import json
import os

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class OffTheDipDataset(CustomDataset):

    CLASSES = ('Pedestrian',)

    studio_label_ann = 'my_project_files/data_training/labels/3.json'

    def load_annotations(self, ann_file):
        with open(self.studio_label_ann) as f:
          d = json.load(f)

        results = d["result"]

        data_infos = []
        frames = []

        for res in results:
            seq = res["value"]["sequence"]
            for f in seq:
                if f["frame"] not in frames:
                    frames.append(f["frame"])
                    data_info = dict(filename=f'{f["frame"]}.jpg', width=1280, height=720, ann=dict(bboxes=[], labels=[], bboxes_ignore=[], labels_ignore=[]))
                    bboxes = [f["x"], f["y"], (f["x"]+f["width"]), (f["y"]+f["height"])]
                    data_info["ann"]["bboxes"].append(bboxes)
                    data_info["ann"]["labels"].append(0)
                    data_infos.append(data_info)

                else:
                    for d in data_infos:
                        if d["filename"] == f'{f["frame"]}.jpg':
                            bboxes = [f["x"], f["y"], (f["x"] + f["width"]), (f["y"] + f["height"])]
                            d["ann"]["bboxes"].append(bboxes)
                            d["ann"]["labels"].append(0)

        for d in data_infos:
            d["ann"]["bboxes"] = np.array(d["ann"]["bboxes"], dtype=np.float32).reshape(-1, 4)
            d["ann"]["labels"] = np.array(d["ann"]["labels"], dtype=np.long)
            d["ann"]["bboxes_ignore"] = np.array(d["ann"]["bboxes_ignore"], dtype=np.float32).reshape(-1, 4)
            d["ann"]["labels_ignore"] = np.array(d["ann"]["labels_ignore"], dtype=np.long)

        return data_infos
