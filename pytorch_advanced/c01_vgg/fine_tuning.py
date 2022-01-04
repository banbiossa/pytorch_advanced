"""for training the fine tuning"""
from __future__ import annotations
import torch
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_model(
    net,
    dataloader_dict: dict,
    criterion,
    optimizer,
    num_epochs: int,
    force_cpu: bool = False,
):
    """train model

    :param net: the network (torchvision.models.vgg.VGG16)
    :param dataloader_dict:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :param force_cpu:
    :return:
    """

    def _get_device():
        if force_cpu:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    device = _get_device()
    logger.info(f"Using {device}")

    # to GPU
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # epoch loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 30)

        # train/val per epoch
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            # skip first epoch
            if (epoch == 0) and (phase == "train"):
                continue

            # mini batch
            for inputs, labels in tqdm(dataloader_dict[phase]):

                # gpu
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            # per epoch
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
