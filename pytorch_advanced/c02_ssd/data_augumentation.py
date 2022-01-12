"""
GitHub：amdegroot/ssd.pytorch を参考にしています。

MIT License
Copyright (c) 2017 Max deGroot, Ellis Brown
https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
data augumentation BBoxごと変形させる
"""
from __future__ import annotations
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    """2つのbox が交差する面積ぽい"""
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.
    The jaccard overlap is simply the intersection over union of two boxes.

    E.g:
        $$ A \cap B / A \cup B = A \cap B / (area(A) + area(B) - A \cap B) $$

    Args:
        box_a: multiple bounding boxes, shape: [num_boxes, 4]
        box_b: single bounding box, shape: [4]

    Returns:
        jaccard overlap, Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A, B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]


class Compose:
    def __init__(self, transforms_list: list[transforms]):
        """Composes several augumentations together

        Args:
            transforms_list: list of transforms to compose

        Examples:
            >>> augumentations.Compose([
            >>>     transforms.CenterCrop(10),
            >>>     transforms.ToTensor(),
            >>> ])
        """
        self.transforms_list = transforms_list

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms_list:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda:
    def __init__(self, lambd):
        """Applies a lambda as a transform

        Args:
            lambd:
        """
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts:
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords:
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape

        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower"
        assert self.lower >= 0, "contrast lower must be non-negative"

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels


class RandomHue:
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class Temp:
    def __call__(self, image, boxes=None, labels=None):
        return image, boxes, labels
