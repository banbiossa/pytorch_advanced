"""
https://github.com/amdegroot/ssd.pytorch
のbox_utils.pyより使用
関数matchを行うファイル

本章の実装はGitHub：amdegroot/ssd.pytorch [4] を参考にしています。
MIT License
Copyright (c) 2017 Max deGroot, Ellis Brown

"""
from __future__ import annotations

import torch
from torch import Tensor


def point_form(boxes) -> Tensor:
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.

    Args:
        boxes: tensor, center-size default boxes from priorbox layers

    Returns: boxes: tensor, converted xmin, ymin, xmax, ymax form of boxes
    """
    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] + boxes[:, 2:] / 2,
            1,
        )
    )


def center_size(boxes) -> Tensor:
    """convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data

    Args:
        boxes: point_form boxes

    Returns:

    """
    return torch.cat(
        (
            boxes[:, :2] + boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] - boxes[:, 2:] / 2,
            1,
        )
    )


def intersect(box_a, box_b) -> Tensor:
    """Resize both tensors to [A, B, 2] without new malloc:
    [A, 2] -> [A, 1, 2] -> [A, B, 2]
    [B, 2] -> [1, B, 2] -> [A, B, 2]
    Then we compute the area of intersect between box_a and box_b

    Args:
        box_a: (tensor) bounding boxes, shape: [A, 4]
        box_b: (tensor) bounding boxes, shape: [B, 4]

    Returns: (tensor) intersection area, shape: [A, B]
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_a[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_a[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b) -> Tensor:
    """Compute the jaccard overlap of two sets of boxes
    Args:
        box_a:
        box_b:

    Returns:

    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]).unsqueeze(
        1
    ).expand_as(inter)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]).unsqueeze(
        1
    ).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(threshold: float, truths, priors, variances, labels, loc_t, conf_t, idx: int):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds

    Args:
        threshold: overlap threshold to match boxes
        truths: (tensor)ground truth boxes, shape: [num_obj, num_priors]
        priors:(tensor) prior boxes from priorbox layers, shape: [n_priors, 4]
        variances: (tensor) variances corresponding to each prior coord, [num_priors, 4]
        labels: (tensor) all the class labels for the image, shape: [num_obj]
        loc_t: (tensor) Tensor to be filled w/ encoded location targets
        conf_t: (tensor) tensor to be filled w/ matched indicies for conf preds
        idx: current batch index

    Returns: the matched indices corresponding to 1) location and 2) confidence preds.
    """
    # jaccard
    overlaps = jaccard(truths, point_form(priors))

    # bipartite matching
    # [1, num_priors] best prior of each ground truth
    best_prior_overlap, best_prior_index = overlaps.max(1, keepdim=True)

    # [1, num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_index = overlaps.max(0, keepdim=True)
    best_truth_index.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_index.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_index, 2)  # ensure best prior

    # todo: refactor index best_prior_index with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_index.size(0)):
        best_truth_index[best_prior_index[j]] = j

    matches = truths[best_truth_index]
    conf = labels[best_truth_index] + 1
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors, 4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes

    Args:
        matched: tensor, corrds of ground truth for each prior in point-form
            shape: [num_priors, 4]
        priors: (tensor) prior boxes in center-offset form
            shape: [num_priors, 4]
        variances: (list[float]) variances of priorboxes

    Returns: encoded boxes (tensor), shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]
