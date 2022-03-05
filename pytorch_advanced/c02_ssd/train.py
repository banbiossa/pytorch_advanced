from .ssd_model import (
    make_data_path_list,
    VOCDataset,
    DataTransform,
    AnnoXML2List,
    od_collate_fn,
    SSD,
    MultiBoxLoss,
)
import torch.utils.data as data
import torch
import torch.nn as nn

rootpath = "./data/VOCdevkit/VOC2012"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_data_path_list(
    rootpath
)

voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(
    train_img_list,
    train_anno_list,
    phase="train",
    transform=DataTransform(input_size, color_mean),
    transform_anno=AnnoXML2List(voc_classes),
)

val_dataset = VOCDataset(
    val_img_list,
    val_anno_list,
    phase="val",
    transform=DataTransform(input_size, color_mean),
    transform_anno=AnnoXML2List(voc_classes),
)

batch_size = 32
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn
)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn
)


dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
ssd_cfg = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 11, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

net = SSD(phase="train", cfg=ssd_cfg)

vgg_weigths = torch.load("./weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weigths)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

print("load complete")


criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
