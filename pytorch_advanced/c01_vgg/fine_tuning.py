"""for training the fine tuning"""
from __future__ import annotations
import torch
import logging

logger = logging.getLogger(__name__)


def train_model(net, dataloader_dict: dict, criterion, optimizer, num_epochs: int):
    """ train model

    :param net:
    :param dataloader_dict:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device}")