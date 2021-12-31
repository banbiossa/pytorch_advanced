from __future__ import annotations

"""
For chapter1, part3 of the book.
Make the code here and import as a library
"""
from pathlib import Path
from PIL import Image
from tqdm import tqdm_notebook as tqdm

import torch
import torch.utils.data as data
from torchvision import transforms
from dotenv import find_dotenv

import logging

logger = logging.getLogger(__name__)


class ImageTransform:
    def __init__(self, resize, mean, std):
        """Image preprocessing. resize -> std.
            on train,augment with RandomResizedCrop and RandomHorizontalFlip.

        Args:
            resize:
            mean:
            std:
        """
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        """

        Args:
            img:
            phase: "train" or "val"

        Returns:
        """
        return self.data_transform[phase](img)


def make_datapath_list(phase="train") -> list[str]:
    """list the images in the datapath

    Args:
        phase: "train" or "val"
    """
    root = Path(find_dotenv()).parent
    data_path = root / "notebooks" / "1_image_classification" / "data"
    target_path = data_path / "hymenoptera_data" / phase  # / "**" / "*.jpg"
    path_list = []

    for path in target_path.glob("**/*.jpg"):
        path_list.append(path)
    logger.info(f"Load {len(path_list)} {phase} files")
    return path_list


def test_make_datapath_list():
    actual = make_datapath_list()
    assert type(actual) == list


class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        """Ant and bee dataset implementing pytorch dataset

        Args:
            file_list: file paths
            transform: preprocessing class
            phase: 'train' or 'val'
        """
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        """Return preprocessed data and label"""
        # get image
        img_path = self.file_list[index]
        img = Image.open(img_path)

        # transform
        img_transformed = self.transform(img, self.phase)

        # parse path string to get label name
        label = img_path.parent.stem

        # numeric
        label_dict = {"ants": 0, "bees": 1}
        return img_transformed, label_dict[label]


def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    """Train model

    Args:
        net:
        dataloader_dict:
        criterion:
        optimizer:
        num_epochs:

    Returns:
    """
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # skip epoch 0 to check result of no-train
            if (epoch == 0) and (phase == "train"):
                continue

            # get data from loader
            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # back propagation
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # iterate
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            # loss per epoch
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
