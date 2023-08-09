import os
import cv2
import glob
from io import StringIO
from collections import namedtuple

import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import utils
from common import common
from tools.writer import writer


class GocatorParser:
    LineRange = namedtuple("LineRange", ["start", "end"], defaults=[0, 0])

    CSV_LINES = {
        "info": LineRange(1, 3),
        "deviceInfo": LineRange(6, 8),
        "meta": LineRange(25, 27),
        "data": LineRange(27, -3),
        # "data": LineRange(27, 278),
    }

    def __init__(self, reversed=False):
        self.reversed = reversed

    def parse_gocator_csv(self, file_path):
        f = open(file_path, "r")
        lines = f.readlines()

        dfs = {}
        for key, lrange in self.CSV_LINES.items():
            content = utils.get_content(lines, lrange.start, lrange.end)
            linesIO = StringIO(content)
            df = pd.read_csv(linesIO, low_memory=False)
            dfs[key] = df

        dfs["data"] = dfs["data"].fillna(np.inf)
        num_rows = dfs["data"].loc[:, "Y\X"].shape[0]
        if self.reversed:
            dfs["data"].loc[np.arange(num_rows), "Y\X"] = (
                dfs["data"].loc[np.arange(num_rows)[::-1], "Y\X"].to_numpy()
            )
            dfs["data"] = dfs["data"].sort_values("Y\X")

        f.close()

        return dfs


class GoCator:
    """Class For GoCator-2375 3D Scanning Sensor

    Abstract Camera Model:

    laser emitter    -----------------------
                      |
                      |            clearance_distance
                      |
                    - -----------------------
                      |
                      |
                      |             1/2 measurement_range
                      |
    calibrated base  ---
                      |             Transform Z
    active center  0 ------------------------
                      |
                      |
                      |             1/2 measurement_range
                      |
                      |
                    + -----------------------

    """

    NUM_COLUMNS_FOR_CURRENT_CUT_DEPTH_ESTIMATION = 100

    def __init__(
        self, data_dir, clearance_distance=650, measurement_range=1350, reversed=False
    ):
        # camera parameterss
        self.cd = clearance_distance
        self.mr = measurement_range
        self.active_center = self.cd + self.mr * 0.5

        self.csvs = sorted(
            glob.glob(os.path.join(data_dir, "*.csv")),
            key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[0]),
        )
        self.parser = GocatorParser(reversed)

        self.center_col_offset = 15.0  # important, big bug, shit show
        # self.center_col_offset = -22.0  # important, big bug, shit show

        self.current_cut_depth_override = None
        # self.current_cut_depth_override = 0  # 16inch
        # self.current_cut_depth_override = 406.4  # 16inch

        self.max_particle_diameter = 76.2  # 3inch

    def base_distance_to_camera(self, dfs):
        z_offset = dfs["meta"]["Z Offset"].values[0]
        return self.active_center + z_offset

    def distance_to_camera(self, dfs, z_value):
        return self.base_distance_to_camera(dfs) - z_value

    def get_sensor_value_from_distance_to_camera(self, dfs, d):
        return self.base_distance_to_camera(dfs) - d

    def get_ticks(self, dfs, axis):
        if axis == 1:
            Xs = np.array([float(x_string) for x_string in dfs["data"].columns[1:]])
            return Xs - self.center_col_offset
        elif axis == 0:
            return np.array([float(y_string) for y_string in dfs["data"].values[:, 0]])

    def center_col_id(self, dfs):
        Xs = self.get_ticks(dfs, 1)

        dists_center = np.abs(Xs - 0)

        return np.argmin(dists_center)

    def current_cut_distance_estimation(self, dfs):
        if self.current_cut_depth_override != None:
            return self.current_cut_depth_override

        center_col = self.center_col_id(dfs)
        total_columns = len(dfs["data"].columns[1:])

        start = max(0, center_col - self.NUM_COLUMNS_FOR_CURRENT_CUT_DEPTH_ESTIMATION)
        end = min(
            total_columns,
            center_col + self.NUM_COLUMNS_FOR_CURRENT_CUT_DEPTH_ESTIMATION,
        )
        target_columns = dfs["data"].iloc[:, start:end].to_numpy()

        mask = target_columns != np.inf

        z_value_mean_per_column = np.array(
            [target_columns[:, i][mask[:, i]].mean() for i in range(end - start)]
        )

        x_distance_to_center = np.abs(self.get_ticks(dfs, 1)[start:end])
        x_amplitude = np.max(self.get_ticks(dfs, 1))

        def normal_disttribution(x, mean, sd):
            prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
            return prob_density

        weights = normal_disttribution(x_distance_to_center, 0.0, x_amplitude)

        return self.distance_to_camera(
            dfs, z_value_mean_per_column.dot(weights) / (weights.sum() + 1e-20)
        )

    def get_output_pixel_len(self, dfs):
        return dfs["info"]["XResolution"].values[0]

    def get_scaled_data(self, dfs, force_width=None, use_interp=False, keep_inf=False):
        data = dfs["data"].to_numpy()[:, 1:].copy().astype(np.float32)
        mask = data == np.inf
        valid = np.logical_not(mask)
        lower_bound = np.percentile(data[valid], 0.1)
        data[mask] = -np.finfo(np.float32).max

        x_resolution = dfs["info"]["XResolution"].values[0]
        y_resolution = dfs["info"]["YResolution"].values[0]

        h, w = data.shape

        xs = self.get_ticks(dfs, 1)
        ys = self.get_ticks(dfs, 0)

        aspect_ratio = (y_resolution * h) / (x_resolution * w)

        target_w = force_width if force_width is not None else w
        target_h = round(target_w * aspect_ratio)

        xx = np.linspace(xs.min(), xs.max(), target_w)
        yy = np.linspace(ys.min(), ys.max(), target_h)

        if use_interp:
            f = interp2d(xs, ys, data, kind="linear")
            out = f(xx, yy)
        else:
            point_x, point_y = np.meshgrid(xs, ys)
            points = np.concatenate(
                [point_y.flatten()[:, np.newaxis], point_x.flatten()[:, np.newaxis]],
                axis=1,
            )
            values = data.flatten()
            grid_x, grid_y = np.meshgrid(xx, yy)
            out = griddata(
                points,
                values,
                (grid_y.flatten(), grid_x.flatten()),
                method="nearest",
            ).reshape([target_h, target_w])
        if keep_inf:
            out[out < lower_bound] = np.inf
        return xx, yy, out

    def get_scaled_image(self, dfs, image):
        x_resolution = dfs["info"]["XResolution"].values[0]
        y_resolution = dfs["info"]["YResolution"].values[0]

        data = dfs["data"].to_numpy()[:, 1:].copy()
        h, w = data.shape

        aspect_ratio = (y_resolution * h) / (x_resolution * w)

        target_h = round(w * aspect_ratio)

        _, ww = image.shape[:2]

        return cv2.resize(image, (ww, target_h))

    @classmethod
    def get_image_from_depths(cls, depths):
        depth = depths.copy()
        depth = depth.astype(np.float32)

        is_valid = depth != np.inf
        v_data = depth[is_valid]
        normalized_data = utils.normalize(v_data)
        normalized_grayscale = (normalized_data * 255).astype(np.uint8)

        depth_img = np.zeros_like(depth, dtype=np.uint8)
        depth_img[is_valid] = normalized_grayscale
        return depth_img

    def visualize(self, dfs, export_path=None):
        _, _, data = self.get_scaled_data(dfs, use_interp=False, keep_inf=True)

        image = GoCator.get_image_from_depths(data)

        if export_path is None:
            plt.clf()
            plt.imshow(image) if len(image.shape) == 3 else plt.imshow(
                image, cmap="gray"
            )
            plt.show()
        else:
            writer.write_image(export_path, image)

        return image

    def data_size(self):
        return len(self.csvs)

    def get(self, i):
        if i >= 0 and i < self.data_size():
            return self.parser.parse_gocator_csv(self.csvs[i])

        return None

    def get_filename(self, i):
        return self.csvs[i]

    def get_multi(self, start, end):
        dfs = None
        data = []

        for i in range(start, end):
            tmp_dfs = self.get(i)
            if tmp_dfs is not None:
                if dfs is None:
                    dfs = tmp_dfs
                data.append(tmp_dfs["data"])

        if len(data) == 0:
            return None

        y_resolution = dfs["info"]["YResolution"].values[0]
        y_column_name = data[0].columns[0]

        for i in range(1, len(data)):
            prev_d = data[i - 1]
            d = data[i]

            prev_ys = np.array([float(s) for s in prev_d.to_numpy()[:, 0]])
            cur_ys = np.array([float(s) for s in d.to_numpy()[:, 0]])
            cur_ys = cur_ys - cur_ys.min() + prev_ys.max() + y_resolution

            cur_ys = np.array([f"{y:.5f}" for y in cur_ys])

            data[i][y_column_name].iloc[:] = cur_ys

        dfs["data"] = pd.concat(data)

        return dfs

    def cluster(self, dfs, method="DBSCAN"):
        xs, ys, data = self.get_scaled_data(dfs)

        xv, yv = np.meshgrid(xs, ys)
        zv = data.flatten()
        h, w = data.shape

        mask = zv < 0

        X = np.concatenate(
            [
                yv.flatten().reshape(-1, 1),
                xv.flatten().reshape(-1, 1),
                zv.reshape(-1, 1),
            ],
            axis=1,
        )[mask]

        x_resolution = dfs["info"]["XResolution"].values[0]

        clustering = DBSCAN(eps=x_resolution * 2, min_samples=2).fit(X)
        # brc = Birch(n_clusters=None)
        # brc.fit(X)
        # print(np.unique(clustering.labels_).shape)
        label = clustering.labels_
        from collections import Counter

        c = Counter(clustering.labels_)
        commons = c.most_common(254)

        label_filtered = np.zeros_like(zv[mask], dtype=np.uint8)

        for i, (l, n) in enumerate(commons):
            m = label == l
            label_filtered[m] = i + 1

        label_img = np.zeros(h * w, dtype=np.uint8)
        label_img[mask] = label_filtered

        cv2.imwrite("lb.png", label_img.reshape(h, w))

    def to_point_cloud(self, dfs, export_path=None, visualize=False):
        xs, ys, data = self.get_scaled_data(dfs)

        xv, yv = np.meshgrid(xs, ys)
        zv = data.flatten()

        mask = zv < 0
        # zv = zv - np.median(zv[mask])

        xyz = np.concatenate(
            [
                xv.flatten().reshape(-1, 1),
                yv.flatten().reshape(-1, 1),
                zv.reshape(-1, 1),
            ],
            axis=1,
        )[mask]

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        if export_path is not None:
            out_path = os.path.join(common.output_path, export_path)
            o3d.io.write_point_cloud(out_path, pcd)
        elif visualize:
            o3d.visualization.draw_geometries([pcd])

        return pcd

    def get_multi_point_clouds(self, start, end):
        dfs = []

        for i in range(start, end + 1):
            tmp_dfs = self.get(i)
            if tmp_dfs is not None:
                dfs.append(tmp_dfs)

        if len(dfs) == 0:
            return None

        y_resolution = dfs[0]["info"]["YResolution"].values[0]
        y_column_name = dfs[0]["data"].columns[0]

        pcds = [self.to_point_cloud(dfs[0])]
        for i in range(1, len(dfs)):
            prev_d = dfs[i - 1]["data"]
            d = dfs[i]["data"]

            prev_ys = np.array([float(s) for s in prev_d.to_numpy()[:, 0]])
            cur_ys = np.array([float(s) for s in d.to_numpy()[:, 0]])
            cur_ys = cur_ys - cur_ys.min() + prev_ys.max() + y_resolution

            cur_ys = np.array([f"{y:.5f}" for y in cur_ys])

            dfs[i]["data"][y_column_name].iloc[:] = cur_ys

            pcds.append(self.to_point_cloud(dfs[i]))

        return utils.merge_pcd(pcds)

    @classmethod
    def fix_depth(cls, depth, denoise=False):
        depth = depth.astype(np.float32)

        is_valid = depth != np.inf
        v_data = depth[is_valid]
        if denoise:
            not_noise = np.logical_and(
                depth >= np.percentile(v_data, 0.5),
                depth <= np.percentile(v_data, 99.5),
            )
            is_valid = not_noise

        v_data = depth[is_valid]

        normalized_data = utils.normalize(v_data)
        normalized_grayscale = (normalized_data * 255).astype(np.uint8)

        depth_img = np.zeros_like(depth, dtype=np.uint8)
        depth_img[is_valid] = normalized_grayscale

        mask = depth_img > 0

        h, w = mask.shape

        filled_mask = utils.remove_small_holes(mask, (h // 10) * (w // 10))
        need_fill = np.logical_xor(filled_mask, mask)

        # utils.visualize(need_fill)

        dst1 = np.zeros_like(depth, dtype=np.int32)
        dst2 = np.zeros_like(depth, dtype=np.int32)
        dst = np.zeros_like(depth, dtype=np.uint8)

        interval = 16
        for i in range(interval):
            start_h = h // interval * i
            end_h = min(start_h + h // interval, h)
            for j in range(interval):
                start_w = w // interval * j
                end_w = min(start_w + w // interval, w)

                dst1[start_h:end_h, start_w:end_w] = cv2.inpaint(
                    depth_img[start_h:end_h, start_w:end_w],
                    need_fill[start_h:end_h, start_w:end_w].astype(np.uint8) * 255,
                    5,
                    cv2.INPAINT_NS,
                )  # cv2.INPAINT_TELEA
            # print(f"inpaint_group_{i+1}/{interval}_finished")
        interval = 16
        for i in range(interval):
            start_h = h // interval * i
            end_h = min(start_h + h // interval, h)
            for j in range(interval):
                start_w = w // interval * j + 50
                end_w = min(start_w + w // interval + 50, w)

                dst2[start_h:end_h, start_w:end_w] = cv2.inpaint(
                    depth_img[start_h:end_h, start_w:end_w],
                    need_fill[start_h:end_h, start_w:end_w].astype(np.uint8) * 255,
                    5,
                    cv2.INPAINT_TELEA,
                )

        dst = np.maximum(dst2, dst1)
        # refill the depth map
        interp_data = (
            dst[need_fill].astype(np.float32) / 255.0 * (v_data.max() - v_data.min())
            + v_data.min()
        )

        depth[need_fill] = interp_data

        return depth

    def get_projected_depth_map_to_line_scan(
        self, line_scan, dfs, cropped=True, use_interp=False
    ):
        xs, _, gocator_data = self.get_scaled_data(
            dfs, use_interp=use_interp, keep_inf=True
        )

        gocator_data_mask = GoCator.get_image_from_depths(gocator_data) > 0
        h, w = gocator_data_mask.shape
        min_obj_size = (w // 100) ** 2
        min_hole_size = 64
        gocator_data_mask = utils.get_clean_mask(
            gocator_data_mask, min_obj_size, min_hole_size
        )

        gocator_data[np.logical_not(gocator_data_mask)] = np.inf

        vy, vx = np.where(gocator_data != np.inf)

        xx = xs[vx].reshape([-1, 1])

        zz = (
            self.distance_to_camera(dfs, gocator_data[vy, vx].reshape([-1, 1]))
            - line_scan.offset
        )
        coords = np.concatenate([xx, np.zeros_like(xx), zz], axis=1)

        oxs = line_scan.transformTo(coords.T)
        oxs = (oxs - line_scan.res // 2) + line_scan.res // 2
        left_w = line_scan.res // 2 - int(np.ceil(oxs.min()))
        right_w = int(np.floor(oxs.max())) - line_scan.res // 2
        half_w = max(left_w, right_w)

        start_w = 0 if left_w >= right_w else half_w - left_w
        end_w = half_w * 2 if right_w >= left_w else left_w + right_w

        left = line_scan.res // 2 - half_w
        right = line_scan.res // 2 + half_w

        prj = np.zeros([h, line_scan.res], dtype=np.float32)

        prj[vy, np.minimum((oxs + 0.5).astype(np.int32), line_scan.res - 1)] = (
            -zz[:, 0] - line_scan.offset
        )

        prj[prj == 0] = np.inf

        # gocator_to_linescan_y_scale = self.get_projected_y_scale(line_scan, dfs)

        if not cropped:
            return prj, left + start_w, left + end_w  # , gocator_to_linescan_y_scale
        else:
            return prj[:, left:right], start_w, end_w  # ,  gocator_to_linescan_y_scale

    def get_projected_y_scale(self, line_scan, dfs):
        current_cut_distance_to_line_scan = (
            self.current_cut_distance_estimation(dfs) - line_scan.offset
        )
        current_pixel_size_x_linescan = (
            line_scan.sw
            / line_scan.f
            * current_cut_distance_to_line_scan
            / line_scan.res
        )
        gocator_x_resolution = dfs["info"]["XResolution"].values[0]

        gocator_to_linescan_y_scale = (
            gocator_x_resolution / current_pixel_size_x_linescan
        )

        return gocator_to_linescan_y_scale

    def get_reprojected_points_from_line_scan(self, line_scan, dfs, prj):
        h, w = prj.shape
        vy, vx = np.where(prj != np.inf)
        us = np.arange(w)[vx][:, np.newaxis]
        vs = np.arange(h)[vy][:, np.newaxis]
        ws = -prj[vy, vx][:, np.newaxis] - line_scan.offset
        uvw = np.concatenate([us, vs, ws], axis=1)

        xyz = line_scan.transformFrom(uvw)
        xyz[:, 1] *= self.get_output_pixel_len(dfs)

        return xyz

    def get_reprojected_points_mat_from_line_scan(self, line_scan, dfs, prj):
        h, w = prj.shape
        vy, vx = np.where(prj != np.inf)
        us = np.arange(w)[vx][:, np.newaxis]
        vs = np.arange(h)[vy][:, np.newaxis]
        ws = -prj[vy, vx][:, np.newaxis] - line_scan.offset
        uvw = np.concatenate([us, vs, ws], axis=1)

        xyz = line_scan.transformFrom(uvw)
        xyz[:, 1] *= self.get_output_pixel_len(dfs)

        mat = np.zeros((h, w, 3)) + np.inf
        mat[vy, vx, :] = xyz

        return mat

    def get_reprojected_depth_map_from_line_scan(self, line_scan, dfs, prj):
        xyz = self.get_reprojected_points_from_line_scan(line_scan, dfs, prj)

        return NotImplemented
