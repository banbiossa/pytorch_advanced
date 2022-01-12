"""Main code for sdd"""
from __future__ import annotations
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def make_data_path_list(rootpath: Path):
    """Make a list of data paths

    Args:
        rootpath:
    Returns: as a list
        - train_img_list
        - train_anno_list
        - val_img_list
        - val_anno_list
    """
    # make path templates
    def img_path(x):
        return rootpath / "JPEGImages" / f"{x}.jpg"

    def anno_path(x):
        return rootpath / "Annotations" / f"{x}.xml"

    # train/test file ids
    train_id_names = rootpath / "ImageSets" / "Main" / "train.txt"
    val_id_names = rootpath / "ImageSets" / "Main" / "val.txt"

    def get_lists(id_names, get_path):
        return [get_path(line.strip()) for line in open(id_names)]

    return [
        get_lists(ids, func)
        for ids in (train_id_names, val_id_names)
        for func in (img_path, anno_path)
    ]


class AnnoXML2List:
    def __init__(self, classes: list[str]):
        """get xml annotation data, normalize and to list

        Args:
            classes: classes
        """
        self.classes = classes

    def __call__(self, xml_path: str, width: int, height: int):
        """get xml annotation data, normalize and to list

        Args:
            xml_path: path to xml
            width:
            height:

        Returns: [[xmin, ymin, xmax, ymax, label_ind], ...]
        """
        ret = []
        xml = ET.parse(xml_path).getroot()
        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            bbox_list = []
            name = obj.find("name").text.lower().strip()
            bndbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                cur_pixel = int(bndbox.find(pt).text) - 1
                if pt in ("xmin", "xmax"):
                    cur_pixel /= width
                else:
                    cur_pixel /= height
                bbox_list.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bbox_list.append(label_idx)

            # res in [xmin, ...] を足す
            ret += [bbox_list]

        return np.array(ret)


def test_order():
    actual = make_order(["one", "two"], ["three", "four"])
    expected = [("one", "three")]
    assert actual[0] == expected[0]


def make_order(first, second):
    return [(a, b) for a in first for b in second]
