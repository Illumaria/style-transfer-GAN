"""
Stylization function
"""

import logging
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

from style_models import load_image, convert_from_tensor, convert_to_bytes, load_model

TO_TENSOR = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

logger = logging.getLogger("styling")


def stylize(content_image: Image, style_model: nn.Module) -> Image:
    """
    NN stylization
    :param content_image: Image to stylize
    :param style_model: model for stylization
    :return: stylized image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = TO_TENSOR(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(content_image).cpu()

    img = convert_from_tensor(output.squeeze(0))
    return img


def stylize_image(file, style: str) -> str:
    """
    Preprocessing and stylization of bytes
    :param file: bytes of image
    :param style: name of style
    :return: stylized image as bytes
    """
    logger.info("Start to load image")
    content_image = load_image(file)
    orig_size = content_image.size
    logger.info("Image is loaded, image size %s, start loading model: %s.pth" % (orig_size, style))
    style_model = load_model(style)
    logger.info("Model is loaded, start stylization with style %s" % style)
    img = stylize(content_image, style_model)
    logger.info("Stylization is completed")
    img = img.resize(orig_size)
    return convert_to_bytes(img)
