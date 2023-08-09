import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from PySide6 import QtGui
from PySide6 import QtCore
from skimage.transform import resize

import cv2


import utils
from camera.gocator import GoCator, GocatorParser
from camera.linescan import LineScan
from gui.qimgutils import qImgUtils
from gui.widgets.lazyCanvas.lazyGraphicsItem import LazyGraphicsPixmapItem
from gui.widgets.lazyCanvas.lazyCanvas import SequentialGraphicsScene
from gui.resourcepool import ResourcePool

from gui.worker import Worker
from tools.logger import default_logger as logger


class GocatorFrame(LazyGraphicsPixmapItem):
    colors = ["black", "blue", "cyan", "green", "yellow", "red", "pink", "white"]
    cvals = [0.0, 0.14, 0.28, 0.41, 0.57, 0.71, 0.86, 1.0]

    def __init__(
        self,
        camera: GoCator,
        frame_id,
        width,
        height,
        original_width,
        original_height,
        parent=None,
    ):
        super().__init__(width, height, original_width, original_height, parent)
        self.camera: GoCator = camera
        self.fid = frame_id
        self.path = camera.get_filename(self.fid)

        self.approx_min_value_limit = None
        self.approx_min_value = None
        self.approx_max_value_limit = None
        self.approx_max_value = None

        self.base_distance_to_camera = None

        self.cached_heatmap = None
        self.high_quality = False
        self.high_quality_output_heatmap = None

        self.threadpool: QtCore.QThreadPool = None
        self.high_quality_worker = None

    def setThreadPool(self, threadpool):
        self.threadpool: QtCore.QThreadPool = threadpool

    def estimate_approx_value_range(self):
        if self.approx_max_value is None or self.approx_min_value is None:
            # TODO: make it in better places to eliminate depulicated calculation
            # the depth here is the raw sensor data, that is, value higher or lower than the calibrated base plane
            self.approx_min_value_limit = np.inf
            self.approx_max_value_limit = -np.inf
            for path in self.camera.csvs:
                dfs = self.camera.parser.parse_gocator_csv(path)
                approx_min_value_limit = (
                    self.camera.base_distance_to_camera(dfs)
                    - self.camera.current_cut_distance_estimation(dfs)
                    - self.camera.max_particle_diameter
                )
                approx_max_value_limit = self.camera.cd

                self.approx_max_value = max(
                    approx_max_value_limit, self.approx_max_value_limit
                )
                self.approx_min_value_limit = min(
                    approx_min_value_limit, self.approx_min_value_limit
                )
        self.approx_max_value_limit = self.approx_max_value
        self.approx_min_value_limit = self.approx_min_value
        return (self.approx_min_value_limit, self.approx_max_value_limit)

    def loadImg(self) -> QtGui.QImage:
        parser: GocatorParser = self.camera.parser
        dfs = parser.parse_gocator_csv(self.path)
        self.base_distance_to_camera = self.camera.base_distance_to_camera(dfs)
        _, _, depth = self.camera.get_scaled_data(dfs, use_interp=True, keep_inf=True)
        depth = depth.astype(np.float32)

        is_valid = depth != np.inf
        v_data = depth[is_valid]

        fit_data = utils.fit_in_range(
            v_data, minv=self.approx_min_value_limit, maxv=self.approx_max_value_limit
        )

        im_np = np.zeros_like(depth, dtype=np.float32)
        im_np[is_valid] = fit_data

        return im_np

    def _prepare_high_quality(self, mask, low_quality, progress_callback):
        h, w, c = low_quality.shape

        mask[mask > 0] = 255
        im_np_gray = cv2.cvtColor(low_quality, cv2.COLOR_RGB2GRAY)
        colsums = im_np_gray.sum(axis=0)
        start = None
        end = None
        for i in range(colsums.shape[0]):
            if colsums[i] / h > 5 and start is None:
                start = i

            if colsums[i] / h < 5 and end is None and start is not None:
                end = i

        mask[:, :start] = 0
        mask[:, end:] = 0

        self.high_quality_output_heatmap = cv2.inpaint(
            low_quality, mask, 3, cv2.INPAINT_TELEA
        )

        return self.high_quality_output_heatmap

    def gen_heatmap_qimg(self, values):
        bounds = utils.fit_in_range(
            np.array([self.approx_min_value, self.approx_max_value]),
            self.approx_min_value_limit,
            self.approx_max_value_limit,
        )
        values_new = utils.fit_in_range(values, bounds[0], bounds[1])
        mask = (values_new == 0).astype(np.uint8) * 255

        norm = plt.Normalize(0.0, 1.0)
        tuples = list(zip(map(norm, self.cvals), self.colors))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)

        im_np = (cmap(values_new)[:, :, :-1].copy(order="C") * 255.0).astype(np.uint8)

        h, w, c = im_np.shape

        im_np = cv2.resize(
            im_np,
            (self.size.width(), round(self.size.width() / self.raw_size.width() * h)),
        )

        mask = cv2.resize(
            mask,
            (
                self.size.width(),
                round(self.size.width() / self.raw_size.width() * h),
            ),
        )

        if np.any(self.cached_heatmap != im_np):  # need to update high quality results
            self.cached_heatmap = im_np.copy()

            if self.threadpool is not None:
                self.high_quality_worker = Worker(
                    self._prepare_high_quality, mask, im_np
                )
                self.high_quality_worker.setAutoDelete(True)
                self.high_quality_worker.signals.finished.connect(
                    self._high_quality_prepared
                )
                self.threadpool.start(self.high_quality_worker)
        elif self.high_quality and self.high_quality_output_heatmap is None:
            if self.threadpool is not None:
                self.high_quality_worker = Worker(
                    self._prepare_high_quality, mask, im_np
                )
                self.high_quality_worker.setAutoDelete(True)
                self.high_quality_worker.signals.finished.connect(
                    self._high_quality_prepared
                )
                self.threadpool.start(self.high_quality_worker)

        if self.high_quality:
            if self.high_quality_output_heatmap is None:
                self.high_quality_output_heatmap = im_np
            return qImgUtils.from_np(
                self.high_quality_output_heatmap, QtGui.QImage.Format_RGB888
            )
        else:
            return qImgUtils.from_np(self.cached_heatmap, QtGui.QImage.Format_RGB888)

    def _high_quality_prepared(self):
        if self.high_quality and self.high_quality_output_heatmap is not None:
            qimage = qImgUtils.from_np(
                self.high_quality_output_heatmap, QtGui.QImage.Format_RGB888
            )
            self.qimage = qimage.scaled(self.size, QtCore.Qt.IgnoreAspectRatio)
            self.preview_config()

    def update(self, resourcepool):
        self.loading = False
        if self is None:  # avoid calling update when item was deleted
            return

        can_update = False
        if resourcepool:
            # print(self.path, self.cvals)
            values = resourcepool.get(str(self.uuid) + "@" + self.path)
            if values is not None:
                qimage = self.gen_heatmap_qimg(values)
                can_update = True
                self.qimage = qimage.scaled(self.size, QtCore.Qt.IgnoreAspectRatio)

                self.preview_config()

        if not can_update:
            self.qpixmap.fill(QtGui.QColor("black"))
            self.setPixmap(self.qpixmap)

    def init_base_distance(self, dfs=None):
        if self.base_distance_to_camera is None:
            dfs = (
                dfs
                if dfs is not None
                else self.camera.parser.parse_gocator_csv(self.path)
            )
            self.base_distance_to_camera = self.camera.base_distance_to_camera(dfs)

    def release(self, resourcepool: ResourcePool = None, erase_pixmap=False):
        return super().release(resourcepool, erase_pixmap, False)

    def update_colormap(self, vmin, vmax):
        self.init_base_distance()
        self.approx_min_value = -(self.base_distance_to_camera - vmin)
        self.approx_max_value = -(self.base_distance_to_camera - vmax)

    def update_quality(self, enable_high_quality):
        if self.high_quality != enable_high_quality:
            self.high_quality = enable_high_quality


class ProjectedGocatorFrame(GocatorFrame):
    def __init__(
        self,
        gocator: GoCator,
        linescan: LineScan,
        frame_id,
        width,
        height,
        original_width,
        original_height,
        parent=None,
    ):
        parser: GocatorParser = gocator.parser
        dfs = parser.parse_gocator_csv(gocator.get_filename(frame_id))
        y_scale = gocator.get_projected_y_scale(linescan, dfs)

        super().__init__(
            gocator,
            frame_id,
            width,
            round(height * y_scale),
            original_width,
            round(original_height * y_scale),
            parent,
        )
        # print(self.path, y_scale)
        self.linescan = linescan
        # self.use_high_quality = False
        self.scale_y = y_scale
        # self.inch2pixel = None
        self.inch2pixel = (
            1.0 / (self.camera.get_output_pixel_len(dfs) / 25.4) * self.scale_y
        )

    def estimate_approx_value_range(self):
        if self.approx_max_value_limit is None or self.approx_min_value is None:
            self.approx_min_value_limit = np.inf
            self.approx_max_value_limit = -np.inf
            for path in self.camera.csvs:
                dfs = self.camera.parser.parse_gocator_csv(path)
                # the depth here is the negative distance to camera
                approx_min_value_limit = (
                    -self.camera.current_cut_distance_estimation(dfs)
                    - self.camera.max_particle_diameter
                )
                approx_max_value_limit = -self.camera.cd

                self.approx_max_value_limit = max(
                    approx_max_value_limit, self.approx_max_value_limit
                )
                self.approx_min_value_limit = min(
                    approx_min_value_limit, self.approx_min_value_limit
                )
        return (self.approx_min_value_limit, self.approx_max_value_limit)

    def loadImg(self) -> QtGui.QImage:
        parser: GocatorParser = self.camera.parser
        dfs = parser.parse_gocator_csv(self.path)
        self.init_base_distance(dfs)
        depth, _, _ = self.camera.get_projected_depth_map_to_line_scan(
            self.linescan,
            dfs,
            use_interp=True,
            cropped=False,
        )
        depth = depth.astype(np.float32)

        is_valid = depth != np.inf
        v_data = depth[is_valid]

        fit_data = utils.fit_in_range(
            v_data, minv=self.approx_min_value_limit, maxv=self.approx_max_value_limit
        )

        im_np = np.zeros_like(depth, dtype=np.float32)
        im_np[is_valid] = fit_data

        im_np_resized = resize(im_np, (im_np.shape[0] * self.scale_y, im_np.shape[1]))

        # return im_np_resized
        return im_np_resized
