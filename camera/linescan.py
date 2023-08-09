import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ExifTags

from common import common
from utils import visualize


class LineScan:
    def __init__(
        self,
        data_dir,
        focus_length=24,
        resolution=4096,
        sensor_width=30,
        # sensor_width=30.72,
        height_offset=155.0,
    ):
        self.f = focus_length
        self.sw = sensor_width
        self.res = resolution
        supported_image_formats = [".bmp", ".png", ".jpg", ".jpeg", ".tiff"]
        files = os.listdir(data_dir)
        image_extension = None
        for file in files:
            fn, ext = os.path.splitext(file)
            if ext.lower() in supported_image_formats:
                image_extension = ext
                break
        assert image_extension is not None
        self.imgs = sorted(glob.glob(os.path.join(data_dir, f"*{image_extension}")))
        self.pixel_len = sensor_width / self.res

        self.K = np.array(
            [
                [self.f / self.pixel_len, 0, self.res // 2],
                [0, self.f / self.pixel_len, 0],
                [0, 0, 1],
            ]
        )
        self.offset = height_offset

    def scale_image(self, cam_height, pixel2mm, image):
        width_mm = self.sw * cam_height / self.f
        target_w = round(width_mm / pixel2mm)

        h, _ = image.shape[:2]
        image = cv2.resize(image, (target_w, h))

        return image

    def data_size(self):
        return len(self.imgs)

    def get(self, i):
        if i >= 0 and i < self.data_size():
            img = cv2.imread(self.imgs[i], cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32) / 255.0
            # img = img[:, ::-1]
            mtime = os.path.getmtime(self.imgs[i])
            return img, mtime

        return None, None

    def get_filename(self, i):
        return self.imgs[i]

    def get_multi(self, start, end):
        data = []
        mtimes = []
        for i in range(start, end):
            d, mtime = self.get(i)
            if d is not None:
                data.append(d)
                mtimes.append(mtime)
        if len(data) == 0:
            return None, None
        return np.concatenate(data, axis=0), mtimes

    def visualize(self, image, export_path=None):
        return visualize(image, export_path)

    def transformTo(self, cam_coords):
        uv = self.K @ cam_coords

        return uv[0, :] / uv[-1, :]

    def transformFrom(self, uvw):
        out = np.zeros_like(uvw)

        # z = w, y = v
        out[:, 1:] = uvw[:, 1:]

        # because u = x * f / w / pixel_len + res//2
        # x = (u-res//2) * pixel_len * w / f

        out[:, 0] = (uvw[:, 0] - self.res // 2) * self.pixel_len * uvw[:, -1] / self.f

        return out
