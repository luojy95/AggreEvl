""" Utilities for numpy image processing """

import numpy as np
from PIL import Image, ImageEnhance


def enhance_brightness(image: Image, factor: float) -> Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def enhance_contrast(image: Image, factor: float) -> Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def enhance_sharpness(image: Image, factor: float) -> Image:
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


kelvin_table = {
    1000: (255, 56, 0),
    1500: (255, 109, 0),
    2000: (255, 137, 18),
    2500: (255, 161, 72),
    3000: (255, 180, 107),
    3500: (255, 196, 137),
    4000: (255, 209, 163),
    4500: (255, 219, 186),
    5000: (255, 228, 206),
    5500: (255, 236, 224),
    6000: (255, 243, 239),
    6500: (255, 249, 253),
    7000: (245, 243, 255),
    7500: (235, 238, 255),
    8000: (227, 233, 255),
    8500: (220, 229, 255),
    9000: (214, 225, 255),
    9500: (208, 222, 255),
    10000: (204, 219, 255),
}


def convert_temperature(image, temperature) -> Image:
    temperature_key = (round(temperature) // 10) * 10
    r, g, b = kelvin_table[temperature_key]
    matrix = (
        r / 255.0,
        0.0,
        0.0,
        0.0,
        0.0,
        g / 255.0,
        0.0,
        0.0,
        0.0,
        0.0,
        b / 255.0,
        0.0,
    )
    return image.convert("RGB", matrix)
