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


class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor:
    def __init__(self, current="BGR", transform="HSV"):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HS)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast must be upper >= lower"
        assert self.lower >= 0, "contrast must be lower >= 0"

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image:
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).tranpose((1, 2, 0)), boxes, labels


class ToTensor:
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class Expand:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = ratio.uniform(0, height*ratio - height)

        expand_image = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top+height), int(left):int(left+width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror:
    def __call__(self, image, boxes, labels):
        _, width = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels


class SwapChannels:
    def __init__(self, swaps):
        """Transforms a tensorized image by swapping the channels in the order
        specified in the swaps tuple

        Args:
            swaps: int triple. final order of channels
                e.g., (2, 1, 0)
        """
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform="HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current="HSV", transform="BGR"),
            RandomContrast(),
        ]
        self.random_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes=None, labels=None):
        im = image.copy()
        im, boxes, labels = self.random_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class RandomSampleCrop:
    def __init__(self):
        self.sample_options = (
            None,  # using entire image
            # sample a patch s.t. MIN jaccard w/ obj in .1, .3, .7, .9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # random
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            pass



class Temp:
    def __call__(self, image, boxes=None, labels=None):
        return image, boxes, labels
