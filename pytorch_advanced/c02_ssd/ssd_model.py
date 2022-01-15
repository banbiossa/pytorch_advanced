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
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from itertools import product
from math import sqrt

from .data_augumentation import (
    Compose,
    ConvertFromInts,
    ToAbsoluteCoords,
    PhotometricDistort,
    Expand,
    RandomSampleCrop,
    RandomMirror,
    ToPercentCoords,
    Resize,
    SubtractMeans,
)
from .match import match

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


class DataTransform:
    def __init__(self, input_size, color_mean):
        """画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
        画像のサイズを300*300 にする。
        学習時はデータオーギュメンテーションをする。

        Args:
            input_size: int, リサイズ先の画像の大きさ
            color_mean: (B, G, R) 各色チャネルの平均値
        """
        self.data_transform = {
            "train": Compose(
                [
                    ConvertFromInts(),
                    ToAbsoluteCoords(),
                    PhotometricDistort(),
                    Expand(color_mean),
                    RandomSampleCrop(),
                    RandomMirror(),
                    ToPercentCoords(),
                    Resize(input_size),
                    SubtractMeans(color_mean),
                ]
            ),
            "val": Compose(
                [
                    ConvertFromInts(),
                    Resize(input_size),
                    SubtractMeans(color_mean),
                ]
            ),
        }

    def __call__(self, img, phase, boxes, labels):
        """

        Args:
            img:
            phase:
            boxes:
            labels:

        Returns:

        """
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        """VOC dataset を作成するクラス。

        Args:
            img_list:
            anno_list:
            phase:
            transform:
            transform_anno:
        """
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """前処理をした画像のテンソル形式のデータとアノテーションを返す"""
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        """前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得"""
        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(str(image_file_path))  # h, w, bgr
        height, width, channels = img.shape

        # 2. xml 形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]
        )

        # 順序を２段階で変更
        # 色チャネルがBGRになっているので、RGBに変更
        # (h, w, channel) -> (色チャネル、高さ、幅）に変更
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # bbox とラベルをセットにした np.array を作成、変数名 gt は ground_truth の意
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, gt, height, width


def od_collate_fn(batch):
    """Dataset から取り出すアノテーションデータのサイズが画像ごとに異なる
    画像内の物体数が2個であれば（2、5）というサイズですが、3個であれば(3,5) などに変化する
    この変化に対応した dataLoader を作成するために、カスタマイズした collate_fn を作成
    collate_fn は pytorch でリストから mini-batch を作成する
    ミニバッチ分の画像が並んでいるリスト変数 batch に
    ミニバッチ番号を指定する次元を先頭に１つ追加して、リストの形を変形する

    Args:
        batch:

    Returns:

    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] は画像img
        targets.append(torch.FloatTensor(sample[1]))  # sample[1]はアノテーションgt
    # imgsはミニバッチサイズのリスト
    # リストの要素は torch.Size([3, 300, 300])
    # このリストを torch.Size([batch_num, 3, 300, 300]) のテンソルに変換
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリスト
    # リストのサイズはミニバッチサイズ
    # list targets の要素は [n, 5]
    # n は画像ごとに異なり、画像内にある物体の数
    # 5 は [xmin, ymin, xmax, ymax, class_index]
    return imgs, targets
