import os
import cv2
import numpy as np
import open3d as o3d


class loader:
    @classmethod
    def load_image(cls, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @classmethod
    def load_point_cloud(cls, path):
        return o3d.io.read_point_cloud(path)

    @classmethod
    def load_data(cls, path):
        _, ext = os.path.splitext(path)

        if ext == ".npy":
            return np.load(path)

        elif ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            return cls.load_image(path)

        elif ext in [".pcd"]:
            return cls.load_point_cloud(path)
