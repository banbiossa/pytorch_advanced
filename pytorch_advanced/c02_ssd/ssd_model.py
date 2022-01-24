"""Main code for sdd"""
from __future__ import annotations

import logging
import random
import xml.etree.ElementTree as ET
from itertools import product
from math import sqrt
from pathlib import Path
from statistics import variance
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as data
from torch import R, Tensor
from torch.autograd import Function
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .data_augumentation import (Compose, ConvertFromInts, Expand, PhotometricDistort,
                                 RandomMirror, RandomSampleCrop, Resize, SubtractMeans,
                                 ToAbsoluteCoords, ToPercentCoords)
from .match import match

patch_typeguard()  # use before @typechecked

logger = logging.getLogger(__name__)

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def make_data_path_list(rootpath: Path) -> list[list[Path]]:
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
    def img_path(x) -> Path:
        return rootpath / "JPEGImages" / f"{x}.jpg"

    def anno_path(x) -> Path:
        return rootpath / "Annotations" / f"{x}.xml"

    # train/test file ids
    train_id_names = rootpath / "ImageSets" / "Main" / "train.txt"
    val_id_names = rootpath / "ImageSets" / "Main" / "val.txt"

    def get_lists(id_names, get_path) -> list[Path]:
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
            "train":
                Compose([
                    ConvertFromInts(),
                    ToAbsoluteCoords(),
                    PhotometricDistort(),
                    Expand(color_mean),
                    RandomSampleCrop(),
                    RandomMirror(),
                    ToPercentCoords(),
                    Resize(input_size),
                    SubtractMeans(color_mean),
                ]),
            "val":
                Compose([
                    ConvertFromInts(),
                    Resize(input_size),
                    SubtractMeans(color_mean),
                ]),
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

    def __init__(
        self,
        img_list: list[Path],
        anno_list: list[Path],
        phase: str,
        transform: Callable,
        transform_anno: Callable,
    ):
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

    def __getitem__(self, index) -> tuple[Tensor, np.ndarray]:
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
        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4],
                                            anno_list[:, 4])

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


def make_vgg():
    """Make the vgg layers"""
    layers = []
    in_channels = 3

    # channels
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "MC",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ]

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "MC":
            # max ceil, default is max floor
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


def make_extras():
    """8層の extras layer"""
    layers = []
    in_channels = 1024

    # extras cfg
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=3)]
    # relu within SSD
    return nn.ModuleList(layers)


# def to_print(i):
#    return f"layers += [nn.Conv2d(cfg[{i}], cfg[{i+1}], kernel_size=3)]"


def make_loc_conf(num_classes=21, bbox_aspect_num=None):
    """デフォルトボックスのオフセットを出力する loc_layers
    デフォルトボックスに対する各クラスの信頼度　confidence を出力する　conf_layers を作成する

    Returns:
    """
    if bbox_aspect_num is None:
        bbox_aspect_num = [4, 6, 6, 6, 4, 4]
    loc_layers = []
    conf_layers = []

    # VGG-22, conv4_3 (source1) の conv
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [
        nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)
    ]

    # vgg の最終層 (source2)に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [
        nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)
    ]

    # extra の (source3)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [
        nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)
    ]

    # extra の (source4)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [
        nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)
    ]

    # extra の (source5)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [
        nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)
    ]

    # extra の (source6)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [
        nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)
    ]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


def to_print(i):
    print(f"# extra の (source{i+1}に対する畳み込み層")
    print(
        f"loc_layers += [nn.Conv2d(512, bbox_aspect_num[{i}] * 4, kernel_size=3, padding=1)]"
    )
    print(
        f"conf_layers += [nn.Conv2d(1024, bbox_aspect_num[{i}] * 4, kernel_size=3, padding=1)]"
    )


class L2Norm(nn.Module):

    def __init__(self, input_channels=512, scale=20):
        """convC4_3からの出力をscale=20のL2Norm で正規化する層

        Args:
            input_channels:
            scale:
        """
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        """set all weights to `scale`"""
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        """38*38 features, take sum of square for 512 channels
        use the 38*38 values to normalize features and mul the weights

        Args:
            x:

        Returns:
        """
        # norm will be torch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # mul the weights: torch.Size([512])
        # result is torch.Size([batch_num, 512, 38, 38])
        weights = self.weight.unsqueeze(0).unisqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x
        return out


class DBox:

    def __init__(self, cfg: dict):
        """default boxes

        Args:
            cfg:
                - input_size
                - feature_maps
                - steps
                - min_sizes
                - max_sizes
        """
        self.image_size = cfg["input_size"]  # 300
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(self.feature_maps)
        self.steps = cfg["steps"]  # [8, 16, ...] DBox pixel size
        self.min_sizes = cfg["min_sizes"]  # [30, 60, ...] small square pixel size
        self.max_sizes = cfg["max_sizes"]  # [60, 111, ...]
        self.aspect_ratios = cfg["aspect_ratios"]

    def make_dbox_list(self):
        """make dbox"""
        mean = []
        # feature_maps: [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # feature size
                # 300 / steps: [8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]

                # dbox center
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect ratio 1 small dbox [cx, cy, width, height]
                # min_sizes: [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect ratio 1 large dbox
                # max_sizes: [60, 111, 162, 213, 264, 315]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # other aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # dbox to tensor: torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        # dbox to [0, 1]
        output.clamp_(max=1, min=0)

        return output


class SSD(nn.Module):

    def __init__(self, phase: str, cfg: dict):
        """network

        Args:
            phase: ['train', 'inference']
            cfg:
                - num_classes
                - bbox_aspect_num
                - <for_dbox>
        """
        super(SSD, self).__init__()
        self.phase = phase  # ['train', 'inference']
        self.num_classes = cfg["num_classes"]

        # ssd
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])

        # dbox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        # up till vgg conf4_3
        for k in range(23):
            x = self.vgg[k](x)
        sources.append(x)

        # extras
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # conv to sources
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # contiguous for using view later
            # l(x), c(x) output is [batch_num, 4 * num_aspects, feature_w, feature_h]
            # permute to [batch_num, feature_w, feature_h, 4 * num_aspects]

        # loc to [batch_num, 34_928]
        # conf to [batch_num, 183_372]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc to [batch_num, 8732, 4]
        # conf to [batch_num, 8732, 21]
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(loc.size(0), -1, self.num_classes)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            return self.detect(*output)
        else:
            return output  # loc, conf, dbox_list


def decode(loc, dbox_list):
    """Using offset, convert dbox to bbox

    Args:
        loc: [8732, 4] offets from ssd
        dbox_list: [8732, 4] dbox info

    Returns: boxes [xmin, ymin, xmax, ymax]
    """
    # dbox: [cx, cy, width, height]
    # loc: [d_cx, d_cy, d_width, d_height]
    boxes = torch.cat(
        (
            dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
            dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2),
        ),
        dim=1,
    )
    # becomes: torch.Size([8732, 4])

    # bbox [cx, cy, width, height] -> [xmin, ymin, xmax, ymax]
    boxes[:, :2] -= boxes[:, 2:] / 2  # to (xmin, ymin)
    boxes[:, 2:] += boxes[:, :2]  # to (xmax, ymax)

    return boxes


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """Non-Maximum suppression. Remove bboxes with overlap.

    Args:
        boxes: [確信度0.01を超えたbbox数, 4] bbox
        scores: [確信度0.01を超えたbbox数] conf
        overlap:
        top_k:

    Returns:
        - keep: list. conf の降順に nms を通過した index が格納
        - count: int. nms を通過した bbox の数
    """
    # return values
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep: torch.Size([確信度閾値を超えたbbox数]), all zero

    # 各bbox の面積を計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # copy boxes. use for IOU later
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # sort scores (small -> big)
    v, idx = scores.sort()

    # take top k
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]  # largest conf

        # keep current, remove non-maximum
        keep[count] = i
        count += 1

        # break if last
        if idx.size(0) == 1:
            break

        # pop idx (we used the largest one)
        idx = idx[:-1]

        ##############################
        # remove bbox with large IOU
        ##############################

        # save variables (up to idx)
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # clamp to current box
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, min=y2[i])

        # resize (decrement) w and h
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clamp した状態で bbox の幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # min to 0
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        inter = tmp_w * tmp_h

        # IOU = intersect / (a + b - intersect)
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        # IoU が overlap より小さい idx のみを残す
        idx = idx[IoU.le(overlap)]
        # -> 大きいものが消去される

    return keep, count


class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        """Detection, forward

        Args:
            conf_thresh:
            top_k:
            nms_thresh:
        """
        self.softmax = nn.Softmax(dim=-1)  # conf をsoftmax で正規化
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        """順伝播

        Args:
            loc_data: [batch_num, 8732, 4]
            conf_data: [batch_num, 8732, num_classes]
            dbox_list: [8732, 4]

        Returns: output: torch.Size([batch_num, 21, 200, 5])
            batch_num, class, which top 200, bbox info
        """
        # get sizes
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        # softmax on conf
        conf_data = self.softmax(conf_data)

        # get output template.
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # resize conf data from [batch_num, 8732, num_classes] to [batch_num, num_classes, 8732]
        conf_preds = conf_data.transpose(2, 1)

        # minibatch
        for i in range(num_batch):
            # 1. loc と dboxから修正した bbox
            decoded_boxes = decode(loc_data[i], dbox_list)

            # copy conf
            conf_scores = conf_preds[i].clone()

            # loop fo reach image class (0 is background)
            for cl in range(1, num_classes):
                # 2. conf の閾値を超えた bbox を取り出す
                # conf の閾値を超えているマスクを作成し、
                # 閾値を超えた conf のインデックスを c_mask として取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # conf_scores: [21, 8732]
                # c_mask: torch.size([8732])

                # scores: torch.Size([閾値を超えたbbox数])
                scores = conf_scores[cl][c_mask]

                # 閾値を超えた conf がない場合何もしない
                if scores.nelement() == 0:
                    continue

                # c_mask を decoded_boxes に適用できるようにサイズ変更
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask: [8732, 4]

                # l_mask を decoded_boxes に適用
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask] で1次元になってしまうので
                # view で（閾値を超えたbbox数, 4) サイズに変形し直す

                # 3. non-maximum suppression を実施
                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                # ids: confの降順にnms を通過した index

                # outputにnmsを抜けた結果を格納
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        return output  # torch.Size([1, 21, 200, 5])


class MultiBoxLoss(nn.Module):

    def __init__(self, jaccard_thresh=.5, neg_pos=3, device="cpu") -> None:
        """損失関数の計算

        Args:
            jaccard_thresh (float, optional): [description]. Defaults to .5.
            neg_pos (int, optional): [description]. Defaults to 3.
            device (str, optional): [description]. Defaults to "cpu".
        """
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.neg_pos = neg_pos
        self.device = device

    # forward 関数の実装
    def forward(self, predictions: tuple[TensorType["batch", "dbox", 4],
                                         TensorType["batch", "dbox", "num_classes"],
                                         TensorType["dbox", 4]],
                targets: TensorType["batch", -1, 5]) -> tuple[TensorType, TensorType]:
        """損失関数の計算

        Args:
            predictions (tuple): SSD net の訓練時の出力(tuple)
                - loc: torch.Size([num_batch, 8732, 4])
                - conf: torch.Size([num_batch, 8732, num_class])
                - dbox_list: torch.Size([8732, 4])

            targets (Tensor): [num_batch, num_objs, 5] 
                5 は正解のアノテーション情報 [xmin, ymin, xmax, ymax, label_ind]
        
        Returns:
        loss_l: Tensor, loc loss
        loss_c: Tensor, conf loss
        """
        # SSDモデルの出力がtupleで返ってくるので、unpackする
        loc_data: TensorType["batch", "dbox", 4]
        conf_data: TensorType["batch", "dbox", "num_classes"]
        dbox_list: TensorType["dbox", 4]
        loc_data, conf_data, dbox_list = predictions

        # 要素数を把握
        num_batch: int = loc_data.size(0)
        num_dbox: int = loc_data.size(1)
        num_classes: int = conf_data.size(2)

        # 損失の計算に使用するものを格納する変数を作成
        # conf_t_label: 各Dboxに一番近い正解のBBoxラベルを格納させる
        # loc_t: 各Dboxに一番近い正解のBBoxの位置情報を格納させる
        conf_t_label: TensorType["batch", "dbox"]
        loc_t: TensorType["batch", "dbox", 4]
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # loc_t と conf_t_label に DBox と正解アノテーションtargetをmatchした結果を上書きする
        # mini batch ごとに処理を行う
        for idx in range(num_batch):
            # 現在のミニバッチの正解アノテーション情報を取得 (bbox)
            truths: TensorType[-1, 4] = targets[idx][:, :-1].to(self.device)
            # ラベル [物体1のラベル、物体2のラベル...]
            labels: TensorType[-1] = targets[idx][:, -1].to(self.device)

            # デフォルトボックスを新たな変数で用意
            dbox: TensorType["dbox", 4] = dbox_list.to(self.device)

            # 関数matchを実行し、loc_t とconf_t_labelの内容を更新する
            # 詳細
            # loc_t: 各Dboxに一番近い正解のBBoxの位置情報が上書きされる
            # conf_t_label: 各Dboxに一番近い正解のBBoxのラベルが上書きされる
            # ただし、一番近いBBoxとjaccard_overlap が0.5より小さい場合は、
            # 正解BBoxのラベルconf_t_label は背景クラスの0とする

            # variance はDbox->BBoxに補正計算する際の係数
            variance = [0.1, 0.2]

            # match関数を実行
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t,
                  conf_t_label, idx)

            # --------------
            # 位置の損失: loss_l を計算
            # Smooth L1 Loss. ただし物体を発見したDboxのオフセットのみを計算
            # --------------
            # 物体を検出したBBoxを取り出すマスクを作成
            pos_mask: TensorType["batch", "dbox"] = conf_t_label > 0

            # pos_maskをloc_dataのサイズに変形
            pos_idx: TensorType["batch", "dbox", 4]
            pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

            # Positive dbox の loc_data と 教師データloc_t を取得
            loc_p: TensorType["dbox", 4] = loc_data[pos_idx].view(-1, 4)
            loc_t: TensorType["dbox", 4] = loc_t[pos_idx].view(-1, 4)

            # 物体を発見したPositive DBoxのオフセット情報 loc_t の損失を計算
            loss_l: TensorType = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

            # --------------
            # クラス予測の損失 loss_c を計算
            # 交差エントロピー損失．ただし背景クラスであるDboxが圧倒的に多いので
            # Hard Negative Miningを行い、物体と背景の比率が1:3になるようにする
            # そこで背景クラスと予想したもののうち、損失が小さいものはクラスの損失から除く
            batch_conf: TensorType["batch_dbox",
                                   "num_classes"] = conf_data.view(-1, num_classes)

            # クラス予測の損失関数を計算 (reduction='none' にして、和を取らず、次元を潰さない)
            conf_t_label_view: TensorType["batch_dbox"] = conf_t_label.view(-1)
            loss_c: TensorType = F.cross_entropy(batch_conf,
                                                 conf_t_label_view,
                                                 reduction="none")

            # --------------
            # これからNegative Dboxのうち、Hard Negative Miningで抽出するものを求めるマスクを作成
            # --------------

            # 物体発見したPositive Dboxの損失を0に
            # (注意) 物体はラベルが1以上になっている．ラベル0は背景

            # ミニバッチごとの物体クラス予測の数
            num_pos: TensorType["batch", "dbox"] = pos_mask.long().sum(1, keepdim=True)
            loss_c: TensorType["batch", "dbox"] = loss_c.view(num_batch, -1)
            #  物体を発見した　Dboxの損失を0に
            loss_c[pos_mask] = 0

            # Hard Negative Miningを行う
            # 各Dboxの損失のお大きさは loss_c の順位である idx_rank を求める
            loss_idx: TensorType["batch", "dbox"]
            idx_rank: TensorType["batch", "dbox"]
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
