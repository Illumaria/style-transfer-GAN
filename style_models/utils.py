"""
Helped function
"""


import base64
import io
import re

import torch
from PIL import Image
from torch import nn

from .transformer_net import TransformerNet


def load_image(file: str) -> Image:
    """
    Converting bytes into image
    :param file: file bytes string in base64 format
    :return: Image class
    """
    img = Image.open(io.BytesIO(base64.b64decode(file)))
    img = img.convert('RGB')
    return rotate_by_orient(img)


def rotate_by_orient(img: Image) -> Image:
    """
    Rotate image by exif orientation tag
    :param img: original image
    :return: rotated image
    """
    orientation = 0x0112

    exif = dict(img.getexif())
    if orientation in exif:
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    return img


def convert_to_bytes(img: Image) -> str:
    """
    Convert Image into bytes
    :param img: Image for encoding
    :return: Base64 encoded image
    """
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return base64.b64encode(img_byte_arr.getbuffer()).decode()


def convert_from_tensor(data: torch.Tensor) -> Image:
    """
    Converting tensor to PIL Image
    :param data: tensor
    :return: PIL Image
    """
    data *= torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    data += torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = (data * 255).clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(img)


def load_model(style: str) -> nn.Module:
    """
    Load model weights
    :param style: name of style to load
    :return: NN model with loaded weight
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model = TransformerNet()
    state_dict = torch.load(f'styles/{style}.pth', map_location=device)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    return style_model
