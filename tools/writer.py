import os
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image

from common import common


class writer:
    @classmethod
    def check_out_dir(cls, outpath):
        out_dir, _ = os.path.split(outpath)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    @classmethod
    def write_image(cls, name, image):
        plt.clf()

        out_path = os.path.join(common.output_path, name)
        cls.check_out_dir(out_path)
        if len(image.shape) == 3:
            plt.imsave(out_path, image)
        else:
            plt.imsave(out_path, image, cmap="gray")

        plt.clf()

        return image

    @classmethod
    def write_images_to_gif(cls, name, frames):
        frames_pil = [Image.fromarray(frame) for frame in frames]

        frame_head = next(iter(frames_pil))

        out_path = os.path.join(common.output_path, name)
        cls.check_out_dir(out_path)

        frame_head.save(
            fp=out_path,
            format="GIF",
            append_images=frames_pil,
            save_all=True,
            duration=200,
            loop=0,
        )

        return frames

    @classmethod
    def write_point_cloud(cls, path, points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        cls.check_out_dir(path)
        o3d.io.write_point_cloud(path, pcd)

        return pcd
