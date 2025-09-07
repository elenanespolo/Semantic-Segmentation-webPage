import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
import matplotlib.patches as mpatches
import io
from PIL import Image
from dataclasses import dataclass
from typing import Tuple
from model.deeplabv2 import get_deeplab_v2
from model.build_bisenet import BiSeNet, BiSeNetErr

@dataclass
class GTA5Label:
    name: str
    ID: int
    color: Tuple[int, int, int]

class GTA5Labels_TaskCV2017():
    road = GTA5Label(name = "road", ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(name = "sidewalk", ID=1, color=(244, 35, 232))
    building = GTA5Label(name = "building", ID=2, color=(70, 70, 70))
    wall = GTA5Label(name = "wall", ID=3, color=(102, 102, 156))
    fence = GTA5Label(name = "fence", ID=4, color=(190, 153, 153))
    pole = GTA5Label(name = "pole", ID=5, color=(153, 153, 153))
    light = GTA5Label(name = "light", ID=6, color=(250, 170, 30))
    sign = GTA5Label(name = "sign", ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(name = "vegetation", ID=8, color=(107, 142, 35))
    terrain = GTA5Label(name = "terrain", ID=9, color=(152, 251, 152))
    sky = GTA5Label(name = "sky", ID=10, color=(70, 130, 180))
    person = GTA5Label(name = "person", ID=11, color=(220, 20, 60))
    rider = GTA5Label(name = "rider", ID=12, color=(255, 0, 0))
    car = GTA5Label(name = "car", ID=13, color=(0, 0, 142))
    truck = GTA5Label(name = "truck", ID=14, color=(0, 0, 70))
    bus = GTA5Label(name = "bus", ID=15, color=(0, 60, 100))
    train = GTA5Label(name = "train", ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(name = "motocycle", ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(name = "bicycle", ID=18, color=(119, 11, 32))
    void = GTA5Label(name = "void", ID=255, color=(0,0,0))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
        void
    ]

import os
def load_model(model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name == "DeepLabV2":
        model = get_deeplab_v2()
        checkpoint_path = os.path.join(base_dir, 'step_2A_DeepLab_Cityscapes_epoch_50.pth')
    elif model_name == "BiSeNet":
        model = BiSeNetErr(19, 'resnet18')
        checkpoint_path = os.path.join(base_dir, 'step_2B_epoch_50.pth')
    elif model_name == "BiSeNetFDA":
        model = BiSeNetErr(19, 'resnet18')
        checkpoint_path = os.path.join(base_dir, 'step_4A_FDA_005_BiSeNet-resnet18_GTA5_FDA_epoch_30.pth')
    elif model_name == "BiSeNetDACS":
        model = BiSeNet(19, 'resnet18')
        checkpoint_path = os.path.join(base_dir, 'step_4B_DACS_paper_augmentation_epoch_30.pth')

    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Se checkpoint è un dict con chiave "model_state_dict", usala
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)

    model.eval()
    return model

device = torch.device('cpu')

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

normalize = v2.Normalize(mean=mean, std=std)

H = 512
W = 1024

custom_augmentation = v2.Compose([
    v2.ToTensor(),
    v2.Resize((H,W))
])

def segmented_image(img: Image, model_name: str) -> Image.Image:
    img = img.convert("RGB")
    model = load_model(model_name)
    model.eval()

    transform = v2.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    img_tensor = custom_augmentation(img_tensor)

    outputs = model(normalize(img_tensor))

    predicted_labels = outputs.argmax(1).cpu()

    color_map = {label.ID: label.color for label in GTA5Labels_TaskCV2017.list_}

    label_map = predicted_labels.squeeze(0).numpy()

    label_defs = GTA5Labels_TaskCV2017.list_
    color_map = {label.ID: label.color for label in label_defs}

    colored_img = np.zeros((H, W, 3), dtype=np.uint8)
    unique_ids = np.unique(label_map)

    for id_ in unique_ids:
        if id_ in color_map:
            colored_img[label_map == id_] = color_map[id_]

    segmented_pil = Image.fromarray(colored_img)

    return segmented_pil

# heroku

