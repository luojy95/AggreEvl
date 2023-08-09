import sys

import PySide6.QtGui

sys.path.append(".")

import os
import copy
import shutil
import time
import yaml
from typing import Optional
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
from typing import NamedTuple
from PIL import Image
from typing import List
from collections import deque
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2
from skimage.transform import resize
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
)

from camera.linescan import LineScan
from camera.gocator import GoCator

from gui.widgets.lazyCanvas.lazyCanvas import LazyCanvas, LazySequentialCanvas
from gui.widgets.lazyCanvas.sequentialPIxmapOverlayCanvas import (
    SequentialPixmapOverlayCanvas,
)
from gui.widgets.lazyCanvas.gocatorFrame import GocatorFrame
from gui.widgets.lazyCanvas.lazyCanvas import SequentialGraphicsScene
from gui.apps.linescanAdjustor.app import LineScanAdjustorMainWindow
from gui.widgets.lazyCanvas.config.common import Range, Crop, Scale
from gui.widgets.lazyCanvas.config.lazyCanvasViewConfig import LazyCanvasViewConfig
from gui.widgets.lazyCanvas.config.lazyGraphicsPixmapItemConfig import (
    LazyGraphicsPixmapItemConfig,
)
from gui.widgets.colorbarLabel import ColorBarLabel
from gui.apps.alignmentTool.config import AlignmentProjectConfig, Point, Rect
from gui.apps.resultVisualizationTool.app import ResultVisualizationMainWindow

# from gui.test.rect import test
from gui.test.lazypixmap import test_projected_gocator, test_linescan, test_gocator
from gui.resourcepool import ResourcePool
from gui.worker import Worker
from tools.imageprocess import *
from utils import to_uint8_img, fit_in_range

from tools.logger import default_logger as logger
from tools.writer import writer

from datetime import datetime

from scipy.interpolate import griddata


class AlignmentWidget(QWidget):
    MINIMUM_WIDTH_CANVAS = 600
    MINIMUM_HEIGHT_CANVAS = 960

    # custom signals
    linescan_load_success: QtCore.Signal = QtCore.Signal(object)

    # default values
    default_calibration_ball_diameter: float = 40

    class EditCheckOnFocusOut(QtWidgets.QLineEdit):
        def __init__(self, focusOutFn):
            super().__init__()
            self.focusOutFn = focusOutFn

        def focusOutEvent(self, arg__1: QtGui.QFocusEvent) -> None:
            super().focusOutEvent(arg__1)
            return self.focusOutFn()

    def __init__(self, resourcepool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout()
        self.pool = resourcepool

        self.project = AlignmentProjectConfig()

        # title font
        title_font = QtGui.QFont()
        title_font.setBold(True)

        canvas_layout = QHBoxLayout()
        overal_btn_layout = QHBoxLayout()

        """Linescan Canvas"""

        linescan_layout = QVBoxLayout()
        self.linescan_label = QtWidgets.QLabel("Line Scan View")
        self.linescan_label.setFont(title_font)
        linescan_layout.addWidget(self.linescan_label)
        self.linescan_canvas = LazySequentialCanvas(self.pool)
        self.linescan_canvas.view.crop_rect_alpha = 255
        self.linescan_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.linescan_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS)
        linescan_layout.addWidget(self.linescan_canvas, 10)
        canvas_layout.addLayout(linescan_layout)
        btn_layout = QVBoxLayout()
        self.flip_btn = QPushButton(text="Horizontally Flip Line Scan Images")
        self.flip_btn.setCheckable(True)
        self.flip_btn.toggled.connect(self.flip_linescan)

        settings_hbox = QHBoxLayout()
        self.reload_settings_label = QtWidgets.QLabel(
            "No Current Line Scan Settings Used."
        )
        self.reload_settings_btn = QPushButton(text="Reload")
        self.reload_settings_btn.clicked.connect(self.reload_settings)
        self.reload_settings_btn.setEnabled(False)
        self.restore_settings_btn = QPushButton(text="Close")
        self.restore_settings_btn.clicked.connect(self.restore_default)
        self.restore_settings_btn.setEnabled(False)
        settings_hbox.addWidget(self.reload_settings_label, 5)
        settings_hbox.addWidget(self.reload_settings_btn, 1)
        settings_hbox.addWidget(self.restore_settings_btn, 1)

        self.adjustor_btn = QPushButton(text="Open Line Scan Adjustor")
        self.adjustor_btn.clicked.connect(self.open_adjustor)

        calibration_ball_edit_layout = QHBoxLayout()
        self.calibration_ball_label = QtWidgets.QLabel(
            "Calibration Ball Diameter (mm):"
        )

        self.calibration_ball_edit = AlignmentWidget.EditCheckOnFocusOut(
            self.checkCalibrationBallText
        )
        self.calibration_ball_edit.setValidator(QtGui.QDoubleValidator(0, 10000, 2))
        self.calibration_ball_edit.setText(
            f"{self.default_calibration_ball_diameter:.2f}"
        )
        self.calibration_ball_edit.setPlaceholderText(
            f"{self.default_calibration_ball_diameter:.2f}"
        )
        calibration_ball_edit_layout.addWidget(self.calibration_ball_label)
        calibration_ball_edit_layout.addWidget(self.calibration_ball_edit)

        self.calibration_ball_btn = QPushButton(text="Select Calibration Ball")
        self.calibration_ball_btn.setCheckable(True)
        self.calibration_ball_btn.toggled.connect(self.toggle_calibration_ball)
        btn_layout.addLayout(settings_hbox)
        btn_layout.addWidget(self.flip_btn)
        btn_layout.addWidget(self.adjustor_btn)
        btn_layout.addLayout(calibration_ball_edit_layout)
        btn_layout.addWidget(self.calibration_ball_btn)
        btn_layout.setContentsMargins(10, 0, 10, 0)
        btn_layout.addStretch()
        overal_btn_layout.addLayout(btn_layout)

        """Gocator Canvas"""
        gocator_layout = QVBoxLayout()
        self.gocator_label = QtWidgets.QLabel("3D Scanner View")
        self.gocator_label.setFont(title_font)
        gocator_layout.addWidget(self.gocator_label)
        self.gocater_canvas = LazySequentialCanvas(self.pool)
        self.gocater_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.gocater_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS)
        gocator_layout.addWidget(self.gocater_canvas, 10)
        canvas_layout.addLayout(gocator_layout)

        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(10, 0, 10, 0)
        btn_layout.addWidget(QtWidgets.QLabel("Height (mm):"))
        self.colorbarLabel = ColorBarLabel(colors=GocatorFrame.colors)
        btn_layout.addWidget(self.colorbarLabel)

        grid_layout = QtWidgets.QGridLayout()
        self.min_dist_label = QtWidgets.QLabel("Min. Height from Alignment Plane (mm):")
        self.min_dist_edit = AlignmentWidget.EditCheckOnFocusOut(
            self.checkColormapMinDistText
        )
        self.min_dist_edit.setValidator(QtGui.QDoubleValidator(-10000, 10000, 2))
        self.min_dist_edit.setText(f"0.00")
        self.min_dist_edit.setPlaceholderText(f"0.00")

        self.max_dist_label = QtWidgets.QLabel("Max. Height from Alignment Plane (mm):")
        self.max_dist_edit = AlignmentWidget.EditCheckOnFocusOut(
            self.checkColormapMaxDistText
        )
        self.max_dist_edit.setValidator(QtGui.QDoubleValidator(-10000, 10000, 2))
        self.max_dist_edit.setText(f"0.00")
        self.max_dist_edit.setPlaceholderText(f"0.00")

        grid_layout.addWidget(self.min_dist_label, 0, 0)
        grid_layout.addWidget(self.min_dist_edit, 0, 1)
        grid_layout.addWidget(self.max_dist_label, 1, 0)
        grid_layout.addWidget(self.max_dist_edit, 1, 1)

        btn_layout.addLayout(grid_layout)

        self.update_colormap_btn = QtWidgets.QPushButton("Update Color Map")
        self.update_colormap_btn.clicked.connect(self.update_colormap_clicked)
        btn_layout.addWidget(self.update_colormap_btn)
        self.gocator_flip_btn = QPushButton(text="Vertically Flip 3D Scanner Images")
        self.gocator_flip_btn.setCheckable(True)
        self.gocator_flip_btn.toggled.connect(self.flip_gocator)
        btn_layout.addWidget(self.gocator_flip_btn)
        self.display_high_quality_btn = QtWidgets.QPushButton("High Quality Color Map")
        self.display_high_quality_btn.setCheckable(True)
        self.display_high_quality_btn.toggled.connect(self.toggle_high_quality)
        btn_layout.addWidget(self.display_high_quality_btn)
        btn_layout.addStretch()

        overal_btn_layout.addLayout(btn_layout)

        """Overlay Canvas"""
        overlay_layout = QVBoxLayout()
        self.overlay_label = QtWidgets.QLabel("Overlay View")
        self.overlay_label.setFont(title_font)
        overlay_layout.addWidget(self.overlay_label)
        self.overlay_canvas = SequentialPixmapOverlayCanvas(self.pool)
        self.overlay_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.overlay_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS)
        overlay_layout.addWidget(self.overlay_canvas, 10)
        canvas_layout.addLayout(overlay_layout)

        self.keypoint_btn = QPushButton(text="Key Point Mode")
        self.keypoint_btn.setCheckable(True)
        self.keypoint_btn.toggled.connect(self.toggle_keypoint_mode)
        self.update_overlay_btn = QPushButton(text="Update Overlay")
        self.update_overlay_btn.clicked.connect(self.update_overlay)
        self.clear_overlay_btn = QPushButton(text="Clear Key Points and Overlay")
        self.clear_overlay_btn.clicked.connect(self.clear_overlay_in_project)

        self.update_overlay_btn.clicked.connect(self.update_overlay)

        analysis_option_label = QtWidgets.QLabel("Analysis Options")
        analysis_option_label.setFont(title_font)
        self.export_point_clouds_btn = QtWidgets.QCheckBox(
            "Export point cloud(s) during analysis."
        )
        self.start_processing_btn = QPushButton(text="Start Analysis")
        self.start_processing_btn.setStyleSheet(
            "QPushButton {background-color:green; color:white;}"
        )
        self.start_processing_btn.clicked.connect(self.start_analysis)
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(10, 0, 10, 0)

        self.alpha_title_label = QtWidgets.QLabel("3D Scanner Image Opacity")
        self.alpha_slider = QtWidgets.QSlider(
            self, orientation=QtCore.Qt.Orientation.Horizontal
        )
        self.alpha_slider.setValue(60)
        self.alpha_label = QtWidgets.QLabel(f"{0.6:.2f}")
        self.alpha_slider.valueChanged.connect(self.adjust_contrast)

        vbox = QVBoxLayout()
        vbox.addWidget(self.alpha_title_label)
        hbox = QHBoxLayout()
        hbox.addWidget(self.alpha_slider)
        hbox.addWidget(self.alpha_label)
        vbox.addLayout(hbox)

        btn_layout.addLayout(vbox)
        btn_layout.addWidget(self.keypoint_btn)
        btn_layout.addWidget(self.clear_overlay_btn)
        btn_layout.addWidget(self.update_overlay_btn)
        btn_layout.addWidget(analysis_option_label)
        btn_layout.addWidget(self.export_point_clouds_btn)
        btn_layout.addWidget(self.start_processing_btn)
        btn_layout.addStretch()

        overal_btn_layout.addLayout(btn_layout)

        layout.addLayout(canvas_layout, 10)
        layout.addLayout(overal_btn_layout)

        self.setLayout(layout)

        self.adjustor = None
        self.visualizer = None

        self.threadpool = QtCore.QThreadPool()
        logger.warning(
            "Multithreading with maximum %d threads" % self.threadpool.maxThreadCount()
        )

        self.current_result_root = None

    def checkCalibrationBallText(self):
        try:
            value = float(self.calibration_ball_edit.text())
            self.calibration_ball_edit.setText(f"{value:.2f}")
            if self.project.calibration_ball_diameter_mm != value:
                self.project.calibration_ball_diameter_mm = value
                self.project.setDirty()
        except:
            self._pop_error_and_log("Invalid calibration ball diameter!")
            self.calibration_ball_edit.setText(
                self.calibration_ball_edit.placeholderText()
            )

    def checkColormapMinDistText(self):
        try:
            value = float(self.min_dist_edit.text())
            value_max = float(self.max_dist_edit.text())
            assert value <= value_max
            self.min_dist_edit.setText(f"{value:.2f}")
        except:
            self._pop_error_and_log("Invalid Minimum Distance!")
            self.min_dist_edit.setText(self.min_dist_edit.placeholderText())

    def checkColormapMaxDistText(self):
        try:
            value = float(self.max_dist_edit.text())
            value_min = float(self.min_dist_edit.text())
            assert value_min <= value

            self.max_dist_edit.setText(f"{value:.2f}")
        except:
            self._pop_error_and_log("Invalid Maximum Distance!")
            self.max_dist_edit.setText(self.max_dist_edit.placeholderText())

    def _pop_error_and_log(self, msg):
        _ = QtWidgets.QMessageBox.critical(
            self,
            "Error",
            msg,
        )
        logger.error(msg)

    def _warning_with_message(self, msg):
        btn = QtWidgets.QMessageBox.warning(
            self,
            "Warning",
            msg,
        )
        logger.error(msg)
        print(btn)

    class LineScanOutputConfig(NamedTuple):
        output_folder: str
        camera: LineScan
        frame_range: Range
        scales: List[Scale]
        brightness: float
        contrast: float
        colortemp: float
        horicrop: Crop
        horiflip: bool
        inch2pixel_ball: float
        inch2pixel_gocator: float
        gocator_linescan_offset: float
        scale_rounding_correction: List[float]
        first_frame_crop: float
        last_frame_crop: float
        export_point_clouds: bool

    class GocatorOutputConfig(NamedTuple):
        output_folder: str
        camera: GoCator
        frame_range: Range
        horicrop: Crop
        vertflip: bool
        cvals: List[float]
        colors: List[str]
        min_displacement_limit: float
        min_displacement: float
        max_displacement_limit: float
        max_displacement: float
        first_frame_crop: float
        last_frame_crop: float
        export_point_clouds: bool

    def start_analysis(self):
        # 0. sanity check
        num_linescan_frames = len(self.linescan_canvas.view.lazyScene().item_seq)
        if num_linescan_frames <= 0:
            self._pop_error_and_log("Empty Project")
            return
        # 1. prepare output path
        output_folder = self.selectOutputFolder()
        if len(output_folder) == 0 or not os.path.exists(output_folder):
            self._pop_error_and_log("Invalid Output Path")
            self.current_result_root = None
            return
        self.current_result_root = output_folder
        logger.info(f"Save output to {output_folder}")

        # 2. fetch configuration from UI thread
        export_point_clouds = self.export_point_clouds_btn.isChecked()
        # a. fetch linescan settings
        linescan_pixmap_config = self.linescan_canvas.view.pixmap_config
        if self.overlay_canvas.bgCanvas != None:
            linescan_frame_range = self.overlay_canvas.cached_range
            linescan_scales = self.overlay_canvas.cached_scales
            gocator_linescan_offset = self.overlay_canvas.cached_offset_x

        else:
            linescan_frame_range = self.linescan_canvas.view.view_config.displayRange
            if not linescan_frame_range.isValid():
                linescan_frame_range = Range(0, num_linescan_frames)
            linescan_scales = [
                self.linescan_canvas.view.view_config.initialScale
                for _ in range(linescan_frame_range.size())
            ]
            gocator_linescan_offset = 0

        linescan_camera = self.linescan_canvas.view.lazyScene().item_seq[0].camera

        scale_rounding_correction = []
        for i in range(linescan_frame_range.start, linescan_frame_range.end):
            height_raw = self.linescan_canvas.scene.item_seq[i].raw_size.height()
            width_raw = self.linescan_canvas.scene.item_seq[i].raw_size.width()
            height_display = self.linescan_canvas.scene.item_seq[i].size.height()
            width_display = self.linescan_canvas.scene.item_seq[i].size.width()
            height_expected = width_display / width_raw * height_raw
            scale_rounding_correction.append(height_display / height_expected)

        inch2pixel_ball = None
        inch2pixel_gocator = None
        if self.linescan_canvas.view.calibration_ball_diameter_pixel is not None:
            try:
                calibration_ball_diameter_mm = float(self.calibration_ball_edit.text())
                calibration_ball_diameter_inch = calibration_ball_diameter_mm / 25.4
                inch2pixel_ball = (
                    self.linescan_canvas.view.calibration_ball_diameter_pixel
                    / calibration_ball_diameter_inch
                ) * linescan_scales[0].x
            except:
                self._pop_error_and_log(
                    f"Invalid calibration ball diameter: {self.calibration_ball_edit.text()}"
                )

        num_gocator_frames = len(self.gocater_canvas.view.lazyScene().item_seq)
        gocator_config = None
        if self.overlay_canvas.aligned_once and num_gocator_frames > 0:
            first_item_y = self.overlay_canvas.cached_display_range.start
            last_item_y = self.overlay_canvas.cached_display_range.end

            print(f"Gocator Range: {(last_item_y - first_item_y) * 4}")

            gocatorScene = self.gocater_canvas.view.lazyScene()
            if isinstance(gocatorScene, SequentialGraphicsScene):
                start_frame_id = max(0, gocatorScene.bisectY(first_item_y) - 1)
                end_frame_id = max(
                    0,
                    min(num_gocator_frames - 1, gocatorScene.bisectY(last_item_y) - 1),
                )

                print(start_frame_id, end_frame_id)

                rate = (
                    gocatorScene.item_seq[start_frame_id].raw_size.width()
                    / gocatorScene.item_seq[start_frame_id].size.width()
                )

                first_crop = (
                    first_item_y - gocatorScene.item_seq[start_frame_id].y()
                ) * rate

                last_crop = (
                    gocatorScene.item_seq[end_frame_id].y()
                    + gocatorScene.item_seq[end_frame_id].size.height()
                    - last_item_y
                ) * rate

                print(start_frame_id, end_frame_id)
                print(
                    first_crop,
                    last_crop,
                    gocatorScene.item_seq[end_frame_id].size.height(),
                )

                inch2pixel_sum = 0
                inch2pixel_num = 0
                for i in range(start_frame_id, end_frame_id + 1):
                    inch2pixel_i = (
                        self.gocater_canvas.view.lazyScene()
                        .item_seq[start_frame_id]
                        .inch2pixel
                    )
                    if inch2pixel_i is not None:
                        inch2pixel_sum += inch2pixel_i
                        inch2pixel_num += 1

                if inch2pixel_num > 0:
                    inch2pixel_gocator = inch2pixel_sum / inch2pixel_num
                first_gocator_frame = self.gocater_canvas.scene.item_seq[0]
                gocator = first_gocator_frame.camera

                max_disp = first_gocator_frame.approx_max_value
                max_disp_limit = first_gocator_frame.approx_max_value_limit
                min_disp = first_gocator_frame.approx_min_value
                min_disp_limit = first_gocator_frame.approx_min_value_limit

                gocator_config = self.GocatorOutputConfig(
                    output_folder=output_folder,
                    camera=gocator,
                    frame_range=Range(start_frame_id, end_frame_id + 1),
                    horicrop=self.overlay_canvas.fg_crop,
                    vertflip=self.gocater_canvas.view.pixmap_config.vertflip,
                    colors=GocatorFrame.colors,
                    cvals=GocatorFrame.cvals,
                    max_displacement_limit=max_disp_limit,
                    max_displacement=max_disp,
                    min_displacement_limit=min_disp_limit,
                    min_displacement=min_disp,
                    first_frame_crop=first_crop,
                    last_frame_crop=last_crop,
                    export_point_clouds=export_point_clouds,
                )

        print(
            f"inch2pixel calibration ball: {inch2pixel_ball}",
            f"inch2pixel calibration gocator: {inch2pixel_gocator}",
        )

        linescan_config = self.LineScanOutputConfig(
            output_folder,
            linescan_camera,
            linescan_frame_range,
            linescan_scales,
            linescan_pixmap_config.brightness,
            linescan_pixmap_config.contrast,
            linescan_pixmap_config.colortemp,
            linescan_pixmap_config.horicrop,
            linescan_pixmap_config.horiflip,
            inch2pixel_ball,
            inch2pixel_gocator,
            gocator_linescan_offset,
            scale_rounding_correction,
            self.overlay_canvas.cached_first_item_crop,
            self.overlay_canvas.cached_last_item_crop,
            export_point_clouds,
        )
        print(linescan_frame_range, linescan_frame_range.size(), len(linescan_scales))
        assert linescan_frame_range.size() == len(linescan_scales)

        print(linescan_pixmap_config.horicrop, linescan_pixmap_config.horiflip)

        # 2. dispatch background long-run tasks
        worker = Worker(self._start_analysis_internal, linescan_config, gocator_config)
        worker.signals.result.connect(self._handle_analysis_output)
        worker.signals.finished.connect(self._finish_analysis)
        worker.signals.progress.connect(self._analysis_progress)

        # Execute
        self.start_processing_btn.setEnabled(False)
        self.threadpool.start(worker)

    def _finish_analysis(self):
        logger.ok("Analysis Worker Closed")
        self.start_processing_btn.setEnabled(True)
        self.open_visualizer()

    def _handle_analysis_output(self, value):
        logger.ok(f"Analysis: {value}")

    def _analysis_progress(self, progress):
        logger.info(f"Analysis Progress: {progress}")

    def _start_analysis_internal(
        self,
        lsconfig: LineScanOutputConfig,
        gconfig: GocatorOutputConfig,
        progress_callback,
    ) -> str:
        # 1. generate output data
        # time.sleep(2)
        # return

        # 0a. prepare gocator output folder
        total_gocator_height = 0
        if gconfig is not None:
            initial_height = 0
            linescan = lsconfig.camera
            gocator_output_folder = os.path.join(lsconfig.output_folder, "gocator")
            if os.path.exists(gocator_output_folder):
                shutil.rmtree(gocator_output_folder)
            os.makedirs(gocator_output_folder)
            gocator_splitted_folder = os.path.join(gocator_output_folder, "splitted")
            os.makedirs(gocator_splitted_folder)
            gocator_highquality_folder = os.path.join(gocator_output_folder, "smooth")
            os.makedirs(gocator_highquality_folder)

            gocator = gconfig.camera
            parser = gocator.parser

            sensor_min = None
            sensor_max = None
            xyzs = []
            for i in range(gconfig.frame_range.start, gconfig.frame_range.end):
                # load data
                dfs = parser.parse_gocator_csv(gocator.get_filename(i))
                if i == gconfig.frame_range.start:
                    sensor_min = gocator.get_sensor_value_from_distance_to_camera(
                        dfs, -gconfig.min_displacement
                    )
                    sensor_max = gocator.get_sensor_value_from_distance_to_camera(
                        dfs, -gconfig.max_displacement
                    )

                # calculate y scale
                y_scale = gocator.get_projected_y_scale(linescan, dfs)
                depth, _, _ = gocator.get_projected_depth_map_to_line_scan(
                    linescan,
                    dfs,
                    use_interp=False,
                    cropped=False,
                )
                depth = depth.astype(np.float32)

                if gconfig.export_point_clouds:
                    # fetch original 3d point data
                    xyz = gocator.get_reprojected_points_mat_from_line_scan(
                        lsconfig.camera, dfs, depth[::-1] if gconfig.vertflip else depth
                    )  # orthogonal
                    print(xyz.shape, depth.shape)
                    h, w, c = xyz.shape
                    ox = np.arange(w)
                    oy = np.arange(h)
                    xx = ox
                    yy = np.linspace(0, h - 1, round(h * y_scale))
                    print(ox.shape, oy.shape, xx.shape, yy.shape)
                    point_x, point_y = np.meshgrid(ox, oy)

                    print(point_x.shape, point_y.shape)
                    points = np.concatenate(
                        [
                            point_y.flatten()[:, np.newaxis],
                            point_x.flatten()[:, np.newaxis],
                        ],
                        axis=1,
                    )
                    print(points.shape)

                    values = xyz.reshape([-1, 3])
                    print(values.shape)
                    grid_x, grid_y = np.meshgrid(xx, yy)
                    xyz = griddata(
                        points,
                        values,
                        (grid_y.flatten(), grid_x.flatten()),
                        method="nearest",
                    ).reshape([round(h * y_scale), w, 3])

                    xyz[:, :, 1] += initial_height
                    initial_height += h * gocator.get_output_pixel_len(dfs)

                    print(xyz.shape, depth.shape)

                is_valid = depth != np.inf
                v_data = depth[is_valid]

                # fit the data in 0-1 range
                fit_data = fit_in_range(
                    v_data,
                    minv=gconfig.min_displacement_limit,
                    maxv=gconfig.max_displacement_limit,
                )

                im_np = np.zeros_like(depth, dtype=np.float32)
                im_np[is_valid] = fit_data

                values = resize(
                    im_np, (round(im_np.shape[0] * y_scale), im_np.shape[1])
                )
                total_gocator_height += round(im_np.shape[0] * y_scale)
                bounds = fit_in_range(
                    np.array([gconfig.min_displacement, gconfig.max_displacement]),
                    gconfig.min_displacement_limit,
                    gconfig.max_displacement_limit,
                )
                values_new = fit_in_range(values, bounds[0], bounds[1])
                mask = (values_new == 0).astype(np.uint8) * 255

                norm = plt.Normalize(0.0, 1.0)
                tuples = list(zip(map(norm, gconfig.cvals), gconfig.colors))
                cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)

                im_np = (cmap(values_new)[:, :, :-1].copy(order="C") * 255.0).astype(
                    np.uint8
                )

                low_quality = im_np.copy()[::-1] if gconfig.vertflip else im_np.copy()

                if i == gconfig.frame_range.start:
                    low_quality = low_quality[round(gconfig.first_frame_crop) :]
                    total_gocator_height -= round(gconfig.first_frame_crop)
                    if gconfig.export_point_clouds:
                        xyz = xyz[round(gconfig.first_frame_crop) :]
                if i == gconfig.frame_range.end - 1:
                    if gconfig.last_frame_crop > 0:
                        low_quality = low_quality[: -round(gconfig.last_frame_crop)]
                        total_gocator_height -= round(gconfig.last_frame_crop)
                        if gconfig.export_point_clouds:
                            xyz = xyz[: -round(gconfig.last_frame_crop)]

                if gconfig.export_point_clouds:
                    xyzs.append(xyz)
                    print(xyz.shape, low_quality.shape)

                if gconfig.horicrop.left > 0:
                    low_quality = low_quality[:, round(gconfig.horicrop.left) :]
                if gconfig.horicrop.right > 0:
                    low_quality = low_quality[:, : -round(gconfig.horicrop.right)]

                image = Image.fromarray(low_quality)

                image_path = os.path.join(
                    gocator_splitted_folder, f"{i-gconfig.frame_range.start+1:04d}.png"
                )
                image.save(image_path)

                h, w, c = im_np.shape

                output_width = 1024
                rate = output_width / w
                output_height = round(h * rate)
                output_size = (output_width, output_height)

                im_np = cv2.resize(im_np, output_size)

                mask = cv2.resize(mask, output_size)

                mask[mask > 0] = 255
                im_np_gray = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
                colsums = im_np_gray.sum(axis=0)
                start = None
                end = None
                h, w, c = im_np.shape

                for j in range(colsums.shape[0]):
                    if colsums[j] / h > 5 and start is None:
                        start = j

                    if colsums[j] / h < 5 and end is None and start is not None:
                        end = j

                mask[:, :start] = 0
                mask[:, end:] = 0

                print(start, end)

                high_quality = cv2.inpaint(im_np, mask, 3, cv2.INPAINT_TELEA)
                high_quality = high_quality[::-1] if gconfig.vertflip else high_quality

                if i == gconfig.frame_range.start:
                    high_quality = high_quality[
                        round(gconfig.first_frame_crop * rate) :
                    ]
                if i == gconfig.frame_range.end - 1:
                    if gconfig.last_frame_crop > 0:
                        high_quality = high_quality[
                            : -round(gconfig.last_frame_crop * rate)
                        ]
                if gconfig.horicrop.left > 0:
                    high_quality = high_quality[
                        :, round(gconfig.horicrop.left * rate) :
                    ]
                if gconfig.horicrop.right > 0:
                    high_quality = high_quality[
                        :, : -round(gconfig.horicrop.right * rate)
                    ]

                image = Image.fromarray(high_quality)

                image_path = os.path.join(
                    gocator_highquality_folder,
                    f"{i-gconfig.frame_range.start+1:04d}.png",
                )
                image.save(image_path)

            logger.info(f"[Analysis]gocator total height: {total_gocator_height}")

            gocator_meta_path = os.path.join(gocator_output_folder, "metadata.yaml")
            try:
                yaml.dump(
                    dict(
                        cvals=gconfig.cvals,
                        colors=gconfig.colors,
                        max_displacement=float(gconfig.max_displacement),
                        max_displacement_limit=float(gconfig.max_displacement_limit),
                        max_height_to_base=float(sensor_max),
                        min_displacement=float(gconfig.min_displacement),
                        min_displacement_limit=float(gconfig.min_displacement_limit),
                        min_height_to_base=float(sensor_min),
                        linescan_align_offset=float(lsconfig.gocator_linescan_offset),
                    ),
                    open(gocator_meta_path, "w+"),
                )
            except:
                logger.error("[AlignmentTool]Failed to output gocator metadata.yaml")

            # return "Gocator Done"
            logger.info("[AlignmentTool]Gocator Info Done.")

            print(lsconfig.gocator_linescan_offset)

        # a. prepare linescan output folder
        linescan_output_folder = os.path.join(lsconfig.output_folder, "linescan")
        if os.path.exists(linescan_output_folder):
            shutil.rmtree(linescan_output_folder)
        os.makedirs(linescan_output_folder)

        # c. apply linescan settings
        linescan_splitted_folder = os.path.join(linescan_output_folder, "splitted")
        os.makedirs(linescan_splitted_folder)
        images = []
        heights = []
        scales = []
        mtimes = []
        width = lsconfig.camera.res - lsconfig.horicrop.left - lsconfig.horicrop.right
        width = round(width * lsconfig.scales[0].x)
        total_height = 0
        last_group_scale = -1
        scale_groups = []
        scale_groups_height_sum = []

        total_height_no_round = 0

        for i in range(lsconfig.frame_range.start, lsconfig.frame_range.end):
            img, mtime = lsconfig.camera.get(i)
            h, w, c = img.shape
            if i == lsconfig.frame_range.start:
                if lsconfig.first_frame_crop > 0:
                    crop_height = min(h - 1, round(lsconfig.first_frame_crop))
                    h -= crop_height
                    img = img[crop_height:]

            if i == lsconfig.frame_range.end - 1:
                if lsconfig.last_frame_crop > 0:
                    crop_height = min(h - 1, round(lsconfig.last_frame_crop))
                    h -= crop_height
                    img = img[:-crop_height]

            heights.append(h)
            scales.append(
                lsconfig.scales[i - lsconfig.frame_range.start].y
                * lsconfig.scale_rounding_correction[i - lsconfig.frame_range.start]
            )
            if scales[-1] != last_group_scale:
                last_group_scale = scales[-1]
                scale_groups.append({"scale": last_group_scale, "data": []})
                scale_groups_height_sum.append({"scale": last_group_scale, "height": 0})
            images.append(to_uint8_img(img, False))
            if heights[-1] > 0 and scales[-1] > 0:
                total_height += round(heights[-1] * scales[-1])
                mtimes.append([total_height, mtime])
            total_height_no_round += heights[-1] * scales[-1]
            scale_groups[-1]["data"].append(i - lsconfig.frame_range.start)
            scale_groups_height_sum[-1]["height"] += heights[-1] * scales[-1]

        logger.info(f"[Analysis]total linescan height: {total_height}")

        np.savetxt(os.path.join(linescan_output_folder, "time.txt"), mtimes)
        all_height = 0

        for gid in range(len(scale_groups)):
            scale = scale_groups[gid]["scale"]
            group = scale_groups[gid]["data"]

            if scale < 0:
                scale_groups[gid]["image"] = None
                continue
            if len(group) > 1:
                group_image = np.concatenate(images[group[0] : group[-1] + 1], axis=0)
            else:
                group_image = images[group[0]]

            if lsconfig.horiflip:
                group_image = group_image[:, ::-1]

            if lsconfig.horicrop.left > 0:
                group_image = group_image[:, lsconfig.horicrop.left :]
            if lsconfig.horicrop.right > 0:
                group_image = group_image[:, : -lsconfig.horicrop.right]

            group_image = Image.fromarray(to_uint8_img(group_image))
            group_image = group_image.resize(
                (width, round(scale_groups_height_sum[gid]["height"]))
            )
            all_height += round(scale_groups_height_sum[gid]["height"])
            group_image = enhance_brightness(group_image, lsconfig.brightness)
            group_image = enhance_contrast(group_image, lsconfig.contrast)
            if lsconfig.colortemp != -1:
                group_image = convert_temperature(group_image, lsconfig.colortemp)

            group_image = np.asarray(group_image)
            scale_groups[gid]["image"] = group_image

        # return

        current_xyz_id = 0
        current_id_row = 0
        current_aligned_height = 0

        final_points = []
        final_colors = []
        pcd_querybook = []
        linescan_offset = round(lsconfig.gocator_linescan_offset)

        def prepare_point_cloud_split(group_image):
            nonlocal current_aligned_height, current_xyz_id, current_id_row, final_points, final_colors, pcd_querybook
            if len(xyzs) > 0 and current_aligned_height < total_gocator_height:
                gh, gw = group_image.shape[:2]
                xyz_w = xyzs[current_xyz_id].shape[1]
                left_crop_linescan = round(
                    lsconfig.horicrop.left * lsconfig.scales[0].x
                )
                left_crop_gocator = round(gconfig.horicrop.left)
                right_crop_gocator = round(gconfig.horicrop.right)

                start_xyz_col = left_crop_gocator
                end_xyz_col = xyz_w - right_crop_gocator

                # start_xyz_col = min(
                #     max(left_crop_gocator, linescan_offset + left_crop_linescan),
                #     xyz_w - 1,
                # )
                # end_xyz_col = min(
                #     gw + linescan_offset + left_crop_linescan,
                #     xyz_w - round(gconfig.horicrop.right),
                # )
                start_line_col = max(
                    0, start_xyz_col - linescan_offset - left_crop_linescan
                )
                end_line_col = min(start_line_col + end_xyz_col - start_xyz_col, gw)

                d1 = end_xyz_col - start_xyz_col
                d2 = end_line_col - start_line_col
                mind = min(d1, d2)
                end_xyz_col = start_xyz_col + mind
                end_line_col = start_line_col + mind

                print(
                    gw,
                    xyz_w,
                    linescan_offset,
                    lsconfig.horicrop,
                    gconfig.horicrop,
                    lsconfig.scales[0].x,
                )

                line_region = group_image[:, start_line_col:end_line_col]
                current_line_height = line_region.shape[0]

                xyz_regions = []
                height_acc = 0
                while (
                    height_acc < current_line_height
                    and current_aligned_height < total_gocator_height
                ):
                    needed = current_line_height - height_acc
                    cur_xyz_region = xyzs[current_xyz_id]
                    cur_remain = xyzs[current_xyz_id].shape[0] - current_id_row
                    if cur_remain > needed:
                        xyz_regions.append(
                            cur_xyz_region[
                                current_id_row : current_id_row + needed,
                                start_xyz_col:end_xyz_col,
                            ]
                        )
                        current_id_row += needed
                        height_acc += needed
                        current_aligned_height += needed

                    elif cur_remain == needed:
                        xyz_regions.append(
                            cur_xyz_region[current_id_row:, start_xyz_col:end_xyz_col]
                        )
                        current_id_row = 0
                        current_xyz_id += 1
                        height_acc += needed
                        current_aligned_height += needed
                    else:
                        xyz_regions.append(
                            cur_xyz_region[current_id_row:, start_xyz_col:end_xyz_col]
                        )
                        current_id_row = 0
                        current_xyz_id += 1
                        height_acc += cur_remain
                        current_aligned_height += cur_remain

                xyz_region = np.concatenate(xyz_regions, axis=0)
                color_region = line_region[:height_acc]
                mask = xyz_region[:, :, 0] != np.inf

                y, x = mask.shape

                gridx, gridy = np.meshgrid(np.arange(x), np.arange(y))
                gridy += current_aligned_height - height_acc
                print(gridy.shape, mask.shape)
                yx = np.concatenate(
                    [gridy[mask].reshape((-1, 1)), gridx[mask].reshape((-1, 1))], axis=1
                )

                print(start_xyz_col, end_xyz_col, start_line_col, end_line_col)
                print(color_region[mask].shape, xyz_region[mask].shape, yx.shape)
                final_points.append(xyz_region[mask])
                final_colors.append(color_region[mask].astype(np.float32) / 255.0)
                pcd_querybook.append(yx)

        print(f"all linescan height: {all_height}")
        current_gid = 0
        queue = deque([])
        queue_height = 0

        image_id = 0
        tmp = 0

        while current_gid < len(scale_groups):
            group_height = round(scale_groups_height_sum[current_gid]["height"])
            if group_height <= 0 or scale_groups[current_gid]["image"] is None:
                current_gid += 1
                continue

            to_use = width - queue_height
            if (
                queue_height + group_height >= width
                and total_height - tmp - to_use >= width
            ):
                queue.append(scale_groups[current_gid]["image"][: width - queue_height])
                group_image = np.concatenate(queue, axis=0)
                if lsconfig.export_point_clouds:
                    prepare_point_cloud_split(group_image)
                image = Image.fromarray(group_image)
                image.save(
                    os.path.join(linescan_splitted_folder, f"{image_id+1:04d}.png")
                )

                image_id += 1
                used_height = width - queue_height
                tmp += used_height
                print(
                    "1",
                    queue_height,
                    group_height,
                    image.height,
                    image.width,
                    tmp,
                    total_height,
                )

                queue.clear()
                queue_height = 0

                remain_height = group_height - used_height
                while remain_height >= width and total_height - tmp - width >= width:
                    group_image = scale_groups[current_gid]["image"][
                        used_height : used_height + width
                    ]
                    if lsconfig.export_point_clouds:
                        prepare_point_cloud_split(group_image)
                    image = Image.fromarray(group_image)
                    image.save(
                        os.path.join(linescan_splitted_folder, f"{image_id+1:04d}.png")
                    )
                    image_id += 1
                    remain_height -= width
                    used_height += width
                    tmp += width

                    print(
                        "2",
                        queue_height,
                        group_height,
                        image.height,
                        image.width,
                        tmp,
                        total_height,
                    )

                if remain_height > 0:
                    queue_height = group_height - used_height
                    queue.append(scale_groups[current_gid]["image"][used_height:])
                    tmp += remain_height
            else:
                queue_height += group_height
                queue.append(scale_groups[current_gid]["image"])
                tmp += group_height

            current_gid += 1

        if len(queue) > 0:
            group_image = np.concatenate(queue, axis=0)
            if lsconfig.export_point_clouds:
                prepare_point_cloud_split(group_image)
            image = Image.fromarray(group_image)
            image.save(os.path.join(linescan_splitted_folder, f"{image_id+1:04d}.png"))
            print(
                "3",
                queue_height,
                group_height,
                image.height,
                image.width,
                tmp,
                total_height,
            )
            queue.clear()
            queue_height = 0

        inch2pixel = (
            lsconfig.inch2pixel_gocator
            if lsconfig.inch2pixel_gocator
            else lsconfig.inch2pixel_ball
        )
        if inch2pixel is None:
            logger.error(f"Cannot Correct Value of Inch2Pixel")
            return "Exit due to no valida inch2pixel"

        logger.info(f"[AlignmentTool]inch2pixel: {inch2pixel}")
        inch2pixel_path = os.path.join(linescan_output_folder, "inch2pixel.yaml")
        try:
            inch2pixel_ball = (
                float(lsconfig.inch2pixel_ball)
                if lsconfig.inch2pixel_ball is not None
                else None
            )
            inch2pixel_gocator = (
                float(lsconfig.inch2pixel_gocator)
                if lsconfig.inch2pixel_gocator is not None
                else None
            )
            yaml.dump(
                dict(
                    inch2pixel_ball=inch2pixel_ball,
                    inch2pixel_gocator=inch2pixel_gocator,
                ),
                open(inch2pixel_path, "w+"),
            )
        except:
            logger.error("[AlignmentTool]Failed to output inch2pixel.yaml")

        if lsconfig.export_point_clouds and len(final_points) > 0:
            logger.info("[Analysis] Start to output splitted Point Cloud")
            for i in range(len(final_points)):
                point_cloud_path = os.path.join(
                    lsconfig.output_folder,
                    "pointclouds",
                    "splitted",
                    f"point_cloud_{i:04d}.ply",
                )
                writer.write_point_cloud(
                    point_cloud_path, final_points[i], final_colors[i]
                )
                point_cloud_querybook_path = os.path.join(
                    lsconfig.output_folder,
                    "pointclouds",
                    "splitted",
                    f"point_cloud_{i:04d}.txt",
                )
                np.savetxt(point_cloud_querybook_path, pcd_querybook[i], fmt="%d")

            logger.info("[Analysis] Start to output combined Point Cloud")
            final_points = np.concatenate(final_points, axis=0)
            final_colors = np.concatenate(final_colors, axis=0)
            pcd_querybook = np.concatenate(pcd_querybook, axis=0)
            point_cloud_path = os.path.join(
                lsconfig.output_folder, "pointclouds", "point_cloud.ply"
            )
            writer.write_point_cloud(point_cloud_path, final_points, final_colors)
            point_cloud_querybook_path = os.path.join(
                lsconfig.output_folder,
                "pointclouds",
                f"point_cloud.txt",
            )
            print(pcd_querybook.max(axis=0), pcd_querybook.min(axis=0))
            np.savetxt(point_cloud_querybook_path, pcd_querybook, fmt="%d")

        # 2. call 2D segmentation
        skip_2d_segmentation = False
        if skip_2d_segmentation:
            logger.warning("[AlignmentTool]2D segmentation skipped")
            return "2D segmentation skipped"

        # log_path = os.path.join(linescan_output_folder, "2DSegmentationLog.txt")
        # if os.path.exists(log_path):
        #     os.remove(log_path)
        # logger.info(
        #     f"[AlignmentTool]Start 2D segmentation in folder {linescan_output_folder} with logging in {log_path}"
        # )
        # cmd = '/bin/bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate i-ballast && '
        # cmd += "cd /home/kelin/Documents/swin && "
        # cmd += "nohup python PROCESS/batch_process.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py cascademaskrcnn_small_1_3batch/epoch_22.pth "
        # cmd += f'--score-thr 0.6 --device cpu --post-process True --path {linescan_output_folder} --inch2pixel {inch2pixel} --half-window-size 1000 > {log_path}"'
        # os.system(cmd)
        logger.info(f"[AlignmentTool]Running 2D segmentation...")
        cmd = '/bin/bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate i-ballast && '
        cmd += "cd /home/kelin/Documents/swin && "
        cmd += "python PROCESS/batch_process.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py cascademaskrcnn_small_1_3batch/epoch_22.pth "
        cmd += f'--score-thr 0.6 --device cpu --path {linescan_output_folder}"'
        os.system(cmd)

        logger.info(f"[AlignmentTool]Generating results from labels...")
        cmd = '/bin/bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate i-ballast && '
        cmd += "cd /home/kelin/Documents/swin && "
        cmd += f'python PROCESS/gen_from_label.py --path {linescan_output_folder} --inch2pixel {inch2pixel} --half_window_size 1000"'
        os.system(cmd)

        # # 3. wrap up
        # logger.info(f"[AlignmentTool]Merging results")
        # cmd = '/bin/bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate i-ballast && '
        # cmd += "cd /home/kelin/Documents/swin && "
        # cmd += f'python PROCESS/update_color_bar.py --path {linescan_output_folder}"'
        # os.system(cmd)

        return "Done."

    def selectOutputFolder(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ".", options
        )
        return path

    def restore_default(self):
        self.reload_settings_label.setToolTip("")
        self.reload_settings_label.setText("No Current Line Scan Settings Used.")
        self.project.linescan_settings_path = None
        self.project.setDirty()
        self.reload_settings_btn.setEnabled(False)
        self.restore_settings_btn.setEnabled(False)
        self._load_settings_internal(
            LazyCanvasViewConfig(), LazyGraphicsPixmapItemConfig()
        )

    def update_colormap_clicked(self):
        return self.update_colormap(True)

    def update_colormap(self, overwrite_project=True):
        try:
            maxv = float(self.max_dist_edit.text())
            minv = float(self.min_dist_edit.text())
            assert minv < maxv
            for frame in self.gocater_canvas.scene.item_seq:
                frame.update_colormap(minv, maxv)
            self.gocater_canvas.view.fresh_visible_items()
            map_cvals = lambda cval: cval * (maxv - minv) + minv
            self.colorbarLabel.values = [map_cvals(cval) for cval in GocatorFrame.cvals]
            self.colorbarLabel.update()
            if overwrite_project:
                if self.project.colormap_start != minv:
                    self.project.colormap_start = minv
                    self.project.setDirty()
                if self.project.colormap_end != maxv:
                    self.project.colormap_end = maxv
                    self.project.setDirty()

        except:
            _ = QtWidgets.QMessageBox.critical(self, "Error", "Invalid Color Map Range")
            logger.error(f"Invalid Color Map Range")

            return

    def load_gocator_data(self, path):
        if not os.path.exists(path) or not os.path.isdir(path):
            _ = QtWidgets.QMessageBox.critical(
                self, "Error", "Invalid 3D Scanner Data folder"
            )
            logger.error(f"Invalid 3D Scanner Data folder: {path}")

        if self.project.linescan_folder is None:
            _ = QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "Cannot load 3D Scanner Data without Line Scan Folder specified",
            )
            logger.error(
                f"Cannot load 3D Scanner Data without Line Scan Folder specified"
            )

        self.overlay_canvas.detachForegroundCanvas()
        self.gocater_canvas.view.cleanupAuxItem()
        self.gocater_canvas.clearSceneItems()

        try:
            vmin, vmax = test_projected_gocator(
                self.gocater_canvas,
                gocator_path=path,
                linescan_path=self.project.linescan_folder,
            )

            self.gocater_canvas.view.initSceneRect()

            metrics = QtGui.QFontMetrics(self.gocator_label.font())
            elidedText = metrics.elidedText(
                f"3D Scanner View: {os.path.relpath(path)}",
                QtCore.Qt.ElideRight,
                self.gocator_label.width(),
            )
            self.gocator_label.setText(elidedText)
            self.gocator_label.setToolTip(path)

            self.min_dist_edit.setPlaceholderText(f"{vmin:.2f}")
            self.min_dist_edit.setText(f"{vmin:.2f}")
            self.min_dist_edit.setValidator(
                QtGui.QDoubleValidator(vmin + 0.01, vmax - 0.01, 2)
            )

            self.max_dist_edit.setPlaceholderText(f"{vmax:.2f}")
            self.max_dist_edit.setText(f"{vmax:.2f}")
            self.min_dist_edit.setValidator(
                QtGui.QDoubleValidator(vmin + 0.01, vmax - 0.01, 2)
            )
            self.gocater_canvas.view.signals.keypoints_removed.connect(
                self._fg_keypoint_removed
            )
            self.gocater_canvas.view.signals.keypoints_set.connect(
                self._fg_keypoint_added
            )
            self.update_colormap(overwrite_project=self.project.gocator_folder != path)
            if self.project.gocator_folder != path:
                self.project.gocator_folder = path
                self.project.setDirty()
                self.clear_overlay_in_project()
            else:
                self.clear_overlay()

        except:
            self.overlay_canvas.detachForegroundCanvas()
            self.gocater_canvas.clearSceneItems()

            _ = QtWidgets.QMessageBox.critical(
                self, "Error", "Failed to open 3D Scanner Data folder"
            )
            logger.error(f"Failed to open 3D Scanner Data folder: {path}")

    def load_linescan_data(self, path):
        self.overlay_canvas.detachBackgroundCanvas()
        self.linescan_canvas.view.cleanupAuxItem()
        self.linescan_canvas.clearSceneItems()
        try:
            test_linescan(self.linescan_canvas, target_path=path)
            self.linescan_canvas.view.initSceneRect()

            metrics = QtGui.QFontMetrics(self.linescan_label.font())
            elidedText = metrics.elidedText(
                f"Line Scan View: {os.path.relpath(path)}",
                QtCore.Qt.ElideRight,
                self.linescan_label.width(),
            )
            self.linescan_label.setText(elidedText)
            self.linescan_label.setToolTip(path)

            self.linescan_canvas.view.signals.keypoints_set.connect(
                self._bg_keypoint_added
            )
            self.linescan_canvas.view.signals.keypoints_removed.connect(
                self._bg_keypoint_removed
            )
            self.linescan_canvas.view.signals.calibration_ball_set.connect(
                self._calibration_ball_add
            )
            if self.project.linescan_folder != path:
                self.project.linescan_folder = path
                self.project.setDirty()
                self.clear_overlay_in_project()
            else:
                self.clear_overlay()
            self.linescan_load_success.emit(path)

        except:
            self.overlay_canvas.detachBackgroundCanvas()
            self.linescan_canvas.clearSceneItems()

            _ = QtWidgets.QMessageBox.critical(
                self, "Error", "Failed to open Line Scan Image folder"
            )
            logger.error(f"Failed to open Line Scan Image folder: {path}")

    def load_linescan_setting(self, path):
        with open(path, "r") as f:
            try:
                data = yaml.safe_load(f)

                displayRange = Range(
                    data["displayRange"]["start"], data["displayRange"]["end"]
                )

                horicrop = (
                    Crop(
                        left=data["horizontalCrop"]["left"],
                        right=data["horizontalCrop"]["right"],
                    )
                    if not self.linescan_canvas.view.pixmap_config.horiflip
                    else Crop(
                        left=data["horizontalCrop"]["right"],
                        right=data["horizontalCrop"]["left"],
                    )
                )
                initscale = Scale(0, data["initialScaleY"])
                colortemp = data["colorTemperature"]
                brightness = data["brightness"]
                contrast = data["contrast"]

                view_config = LazyCanvasViewConfig(displayRange, initscale)

                pixmap_config = LazyGraphicsPixmapItemConfig(
                    brightness, contrast, colortemp, horicrop
                )

                self._load_settings_internal(view_config, pixmap_config)

                metrics = QtGui.QFontMetrics(self.reload_settings_label.font())
                elidedText = metrics.elidedText(
                    f"Current Line Scan Settings: {os.path.split(path)[-1]}",
                    QtCore.Qt.ElideRight,
                    self.reload_settings_label.width(),
                )
                self.reload_settings_label.setToolTip(path)
                self.reload_settings_label.setText(elidedText)
                self.reload_settings_btn.setEnabled(True)
                self.restore_settings_btn.setEnabled(True)
                if self.project.linescan_settings_path != path:
                    self.project.linescan_settings_path = path
                    self.project.setDirty()
                    self.clear_overlay_in_project()
                else:
                    self.clear_overlay()
            except yaml.YAMLError as exc:
                print(exc)

    def _load_settings_internal(
        self,
        view_config: LazyCanvasViewConfig,
        pixmap_config: LazyGraphicsPixmapItemConfig,
    ):
        view_range = self.linescan_canvas.scene.maskDisplayWithRange(
            view_config.displayRange
        )
        current_scene_rect = self.linescan_canvas.view.sceneRect()
        self.linescan_canvas.view.setSceneRect(
            current_scene_rect.x(),
            view_range.start,
            current_scene_rect.width(),
            view_range.end - view_range.start,
        )
        if view_range.start != current_scene_rect.top():
            self.linescan_canvas.view.centerOn(current_scene_rect.x(), view_range.start)
        elif view_range.end - view_range.start != current_scene_rect.height():
            self.linescan_canvas.view.centerOn(current_scene_rect.x(), view_range.end)
        self.linescan_canvas.view.view_config.displayRange = view_config.displayRange
        self.linescan_canvas.view.update_scale_y(view_config.initialScale.y)

        pixmap_config.horiflip = self.linescan_canvas.view.pixmap_config.horiflip
        self.linescan_canvas.view.update_pixmap_config(pixmap_config)
        self.linescan_canvas.view.update_crop_rect()

        self.overlay_canvas.view.fresh_visible_items()

    def open_adjustor(self):
        if self.adjustor is None:
            self.adjustor = LineScanAdjustorMainWindow(self.pool)
        if self.project.linescan_folder is not None:
            self.adjustor.mainWidget.load_data(self.project.linescan_folder)
        if self.project.linescan_settings_path is not None:
            self.adjustor.mainWidget.load_settings(self.project.linescan_settings_path)
        self.adjustor.show()

    def open_visualizer(self):
        if self.visualizer is None:
            self.visualizer = ResultVisualizationMainWindow(self.pool)
        if self.current_result_root is not None:
            self.visualizer.mainWidget.loadResults(self.current_result_root)
        self.visualizer.show()

    def reload_settings(self):
        return self.load_linescan_setting(self.project.linescan_settings_path)

    def update_overlay(self):
        if self.overlay_canvas.fgCanvas is None or self.overlay_canvas.bgCanvas is None:
            self.overlay_canvas.detachForegroundCanvas()
            self.overlay_canvas.detachBackgroundCanvas()
            self.overlay_canvas.setForegroundCanvas(self.gocater_canvas)
            self.overlay_canvas.setBackgroundCanvas(self.linescan_canvas)
            self.overlay_canvas.requestRenderOverlay()
        fg_kps = self.gocater_canvas.view.kp_map
        bg_kps = self.linescan_canvas.view.kp_map
        self.overlay_canvas.alignOverlayWithKeyPoints(fg_kps, bg_kps)

    def clear_overlay_in_project(self):
        self.project.calibration_ball = None
        self.project.keypointsbg.clear()
        self.project.keypointsfg.clear()
        self.project.setDirty()
        self.clear_overlay()

    def clear_overlay(self):
        self.linescan_canvas.view.cleanupCalibrationBall()
        self.linescan_canvas.view.cleanupKeyPoints()
        self.gocater_canvas.view.cleanupKeyPoints()
        self.overlay_canvas.detachForegroundCanvas()
        self.overlay_canvas.detachBackgroundCanvas()

    def adjust_contrast(self, value):
        self.overlay_canvas.adjustForeGroundAlpha(value / 100.0)
        self.alpha_label.setText(f"{value/100.0:.2f}")

    def toggle_high_quality(self):
        if self.display_high_quality_btn.isChecked():
            self.display_high_quality_btn.setStyleSheet(
                "QPushButton {background-color:lightgreen}"
            )
            for frame in self.gocater_canvas.scene.item_seq:
                frame.setThreadPool(self.threadpool)
                frame.update_quality(enable_high_quality=True)
            self.gocater_canvas.view.fresh_visible_items()
        else:
            self.display_high_quality_btn.setStyleSheet("")
            for frame in self.gocater_canvas.scene.item_seq:
                frame.setThreadPool(None)
                frame.update_quality(enable_high_quality=False)
            self.gocater_canvas.view.fresh_visible_items()

    def toggle_keypoint_mode(self):
        if self.calibration_ball_btn.isChecked():
            self.calibration_ball_btn.setChecked(False)
            self.toggle_calibration_ball()
        if self.keypoint_btn.isChecked():
            self.keypoint_btn.setStyleSheet("QPushButton {background-color:lightgreen}")
            self.gocater_canvas.view.enableKeyPointMode(True)
            self.linescan_canvas.view.enableKeyPointMode(True)
        else:
            self.keypoint_btn.setStyleSheet("")
            self.gocater_canvas.view.enableKeyPointMode(False)
            self.linescan_canvas.view.enableKeyPointMode(False)

    def toggle_calibration_ball(self):
        if self.keypoint_btn.isChecked():
            self.keypoint_btn.setChecked(False)
            self.toggle_keypoint_mode()
        if self.calibration_ball_btn.isChecked():
            self.calibration_ball_btn.setStyleSheet(
                "QPushButton {background-color:lightgreen}"
            )
            self.linescan_canvas.view.is_calibration_ball_mode_enabled = True
        else:
            self.calibration_ball_btn.setStyleSheet("")
            self.linescan_canvas.view.is_calibration_ball_mode_enabled = False

    # TODO:
    # def flip_linescan_by_btn(self):
    #     if len(self.project.keypointsbg) > 0 or len(self.project.keypointsfg) > 0:
    #         self._warning_with_message(
    #             "Flip Line Scan Image will lose all key points! Are you sure to continue?"
    #         )

    def flip_linescan(self):
        self.clear_overlay()
        pixmap_config = copy.deepcopy(self.linescan_canvas.view.pixmap_config)
        if self.flip_btn.isChecked():
            self.flip_btn.setStyleSheet("QPushButton {background-color:lightgreen}")
        else:
            self.flip_btn.setStyleSheet("")

        if self.project.linescan_flipped != self.flip_btn.isChecked():
            self.project.linescan_flipped = self.flip_btn.isChecked()
            self.project.setDirty()

        tmp = pixmap_config.horicrop.left
        pixmap_config.horicrop.left = pixmap_config.horicrop.right
        pixmap_config.horicrop.right = tmp
        pixmap_config.horiflip = self.flip_btn.isChecked()

        self.linescan_canvas.view.update_pixmap_config(pixmap_config)
        self.overlay_canvas.view.fresh_visible_items()

    def flip_gocator(self):
        self.clear_overlay()
        pixmap_config = copy.deepcopy(self.gocater_canvas.view.pixmap_config)
        if self.gocator_flip_btn.isChecked():
            self.gocator_flip_btn.setStyleSheet(
                "QPushButton {background-color:lightgreen}"
            )
        else:
            self.gocator_flip_btn.setStyleSheet("")

        if self.project.gocator_flipped != self.gocator_flip_btn.isChecked():
            self.project.gocator_flipped = self.gocator_flip_btn.isChecked()
            self.project.setDirty()

        pixmap_config.vertflip = self.gocator_flip_btn.isChecked()

        self.gocater_canvas.view.update_pixmap_config(pixmap_config)
        self.overlay_canvas.view.fresh_visible_items()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.keypoint_btn.setChecked(False)
            self.calibration_ball_btn.setChecked(False)
            return

        if (
            event.keyCombination().keyboardModifiers()
            == event.keyCombination().keyboardModifiers().ControlModifier
        ):
            if event.key() == QtCore.Qt.Key_K:
                return self.keypoint_btn.setChecked(True)

        if (
            event.keyCombination().keyboardModifiers()
            == event.keyCombination().keyboardModifiers().ControlModifier
        ):
            if event.key() == QtCore.Qt.Key_B:
                return self.calibration_ball_btn.setChecked(True)

        return super().keyPressEvent(event)

    def cleanup(self) -> None:
        self.clear_overlay()
        self.overlay_canvas.view.cleanupConfigs()
        self.gocater_canvas.clearSceneItems()
        self.gocater_canvas.view.cleanupConfigs()
        self.linescan_canvas.clearSceneItems()
        self.linescan_canvas.view.cleanupConfigs()
        self.reload_settings_label.setToolTip("")
        self.reload_settings_label.setText("No Current Line Scan Settings Used.")
        self.gocator_label.setToolTip("")
        self.gocator_label.setText("3D Scanner View")
        self.linescan_label.setToolTip("")
        self.linescan_label.setText("Line Scan View")
        self.reload_settings_btn.setEnabled(False)
        self.restore_settings_btn.setEnabled(False)
        return super().show()

    def _fg_keypoint_added(self, values):
        id, center = values
        self.project.keypointsfg[id] = Point(center.x(), center.y())
        self.project.setDirty()

    def _fg_keypoint_removed(self, values):
        id, center = values
        self.project.keypointsfg.pop(id)
        self.project.setDirty()

    def _bg_keypoint_added(self, values):
        id, center = values
        self.project.keypointsbg[id] = Point(center.x(), center.y())
        self.project.setDirty()

    def _bg_keypoint_removed(self, values):
        id, center = values
        self.project.keypointsbg.pop(id)
        self.project.setDirty()

    def _calibration_ball_add(self, rect: QtCore.QRectF):
        self.project.calibration_ball = Rect(
            rect.x(), rect.y(), rect.width(), rect.height()
        )
        self.project.setDirty()

    def prepare_project(self):
        if self.project.linescan_folder is not None:
            self.load_linescan_data(self.project.linescan_folder)
            if self.project.linescan_settings_path is not None:
                self.load_linescan_setting(self.project.linescan_settings_path)

            if self.project.gocator_folder is not None:
                self.load_gocator_data(self.project.gocator_folder)
                self.gocator_flip_btn.setChecked(self.project.gocator_flipped)

            self.flip_btn.setChecked(self.project.linescan_flipped)

            if (
                self.project.colormap_start is not None
                and self.project.colormap_end is not None
            ):
                self.min_dist_edit.setText(f"{self.project.colormap_start:.2f}")
                self.max_dist_edit.setText(f"{self.project.colormap_end:.2f}")
                self.update_colormap()

            if len(self.project.keypointsfg) > 0:
                self.gocater_canvas.view.overwriteKpPointsWithoutEmit(
                    self.project.keypointsfg
                )
            if len(self.project.keypointsbg) > 0:
                self.linescan_canvas.view.overwriteKpPointsWithoutEmit(
                    self.project.keypointsbg
                )
            if self.project.calibration_ball is not None:
                self.linescan_canvas.view.overwriteCalibrationBallWithoutEmit(
                    self.project.calibration_ball
                )

            self.calibration_ball_edit.setText(
                f"{self.project.calibration_ball_diameter_mm:.2f}"
            )

    def saveProjectSettings(self):
        if self.project.output_path != None:
            if self.project.to_yaml(self.project.output_path):
                self.project.setDirty(False)
                return True
            else:
                return False
        else:
            return self.saveProjectSettingsAs()

    def saveProjectSettingsAs(self) -> bool:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fn, ext = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Alignment Project As ...",
            "alignment_project",
            ".yaml",
            options=options,
        )
        if self.project.to_yaml(fn + ext):
            self.project.setDirty(False)
            return True
        else:
            return False

    def loadProjectSettings(self):
        if self.project.isDirty():
            logger.warning("Project is not saved!")
            if self.project.output_path is None:
                btn = QtWidgets.QMessageBox.warning(
                    self,
                    "Changes not saved",
                    "Create new project?",
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel,
                    defaultButton=QtWidgets.QMessageBox.Cancel,
                )

                if btn == QtWidgets.QMessageBox.Cancel:
                    return False
                elif btn == QtWidgets.QMessageBox.Save:
                    if not self.saveProjectSettingsAs():
                        return False

            else:
                btn = QtWidgets.QMessageBox.warning(
                    self,
                    "Changes not saved",
                    f"Save to {self.project.output_path}?",
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel,
                    defaultButton=QtWidgets.QMessageBox.Cancel,
                )
                if btn == QtWidgets.QMessageBox.Cancel:
                    return False
                elif btn == QtWidgets.QMessageBox.Save:
                    if not self.saveProjectSettings():
                        return False
        self.project.setDirty(False)
        self.cleanup()

        self.project = AlignmentProjectConfig()
        self.project.setDirty(False)

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fn, filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Alignment Project",
            ".",
            "*.yaml",
            options=options,
        )
        self.project.from_yaml(fn)
        self.prepare_project()

    def quitting(self):
        if self.project.isDirty():
            logger.warning("Project is not saved!")
            if self.project.output_path is None:
                btn = QtWidgets.QMessageBox.warning(
                    self,
                    "Changes not saved",
                    "Create new project?",
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel,
                    defaultButton=QtWidgets.QMessageBox.Cancel,
                )

                if btn == QtWidgets.QMessageBox.Cancel:
                    return False
                elif btn == QtWidgets.QMessageBox.Save:
                    if not self.saveProjectSettingsAs():
                        return False

            else:
                btn = QtWidgets.QMessageBox.warning(
                    self,
                    "Changes not saved",
                    f"Save to {self.project.output_path}?",
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel,
                    defaultButton=QtWidgets.QMessageBox.Cancel,
                )
                if btn == QtWidgets.QMessageBox.Cancel:
                    return False
                elif btn == QtWidgets.QMessageBox.Save:
                    if not self.saveProjectSettings():
                        return False
        self.cleanup()
        return True


class AlignmentMainWindow(QMainWindow):
    def __init__(self, resourcepool, parent=None) -> None:
        super().__init__(parent)
        self.pool = resourcepool
        self.mainWidget = AlignmentWidget(self.pool)
        self.setCentralWidget(self.mainWidget)

        self._createMenuBar()
        self.setWindowTitle("I-BALLAST: Alignment Tool")

    def _createMenuBar(self):
        menuBar = QtWidgets.QMenuBar(self)

        loadMenu = QtWidgets.QMenu("&Load", self)

        self.openAlignmentProjectAction = QtGui.QAction("Open Alignment Project", self)
        self.openAlignmentProjectAction.triggered.connect(
            self.mainWidget.loadProjectSettings
        )
        loadMenu.addAction(self.openAlignmentProjectAction)

        loadMenu.addSeparator()

        self.openLineScanAction = QtGui.QAction("Open Line Scan Image Folder", self)
        self.openLineScanAction.triggered.connect(self.openLineScanFolder)
        loadMenu.addAction(self.openLineScanAction)

        self.loadSettingsAction = QtGui.QAction("Load Line Scan Settings", self)
        self.loadSettingsAction.triggered.connect(self.loadLineScanSettings)
        loadMenu.addAction(self.loadSettingsAction)
        self.loadSettingsAction.setDisabled(True)
        self.loadSettingsAction.setVisible(False)

        loadMenu.addSeparator()

        self.openGocatorAction = QtGui.QAction("Open 3D Scanner Image Folder", self)
        self.openGocatorAction.triggered.connect(self.openGocatorFolder)
        loadMenu.addAction(self.openGocatorAction)
        self.openGocatorAction.setDisabled(True)
        self.openGocatorAction.setVisible(False)
        self.mainWidget.linescan_load_success.connect(self._lineScanDataLoaded)

        saveMenu = QtWidgets.QMenu("&Save", self)

        self.saveAlignmentProjectAction = QtGui.QAction("Save Alienment Project", self)
        self.saveAlignmentProjectAction.triggered.connect(
            self.mainWidget.saveProjectSettings
        )
        self.saveAlignmentProjectAsAction = QtGui.QAction(
            "Save Alienment Project as...", self
        )
        self.saveAlignmentProjectAsAction.triggered.connect(
            self.mainWidget.saveProjectSettingsAs
        )
        saveMenu.addAction(self.saveAlignmentProjectAction)
        saveMenu.addAction(self.saveAlignmentProjectAsAction)

        menuBar.addMenu(loadMenu)
        menuBar.addMenu(saveMenu)
        self.setMenuBar(menuBar)

    def _lineScanDataLoaded(self, path):
        self.loadSettingsAction.setVisible(True)
        self.loadSettingsAction.setEnabled(True)
        self.openGocatorAction.setVisible(True)
        self.openGocatorAction.setEnabled(True)

    def openGocatorFolder(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open 3D Scanner Folder", ".", options
        )
        self.mainWidget.load_gocator_data(path)

    def openLineScanFolder(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Line Scan Image Folder", ".", options
        )
        self.mainWidget.load_linescan_data(path)

    def loadLineScanSettings(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fn, filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Line Scan Settings",
            ".",
            "*.yaml",
            options=options,
        )
        self.mainWidget.load_linescan_setting(fn)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.mainWidget.quitting():
            event.accept()
        else:
            event.ignore()


class AlignmentTool(QApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ResourcePool()

        self.mainWindow: AlignmentMainWindow = AlignmentMainWindow(self.pool)

        self.aboutToQuit.connect(self.cleanup)

    def exec(self):
        self.mainWindow.show()
        super().exec()

    def cleanup(self):
        self.pool.terminate_proxy()
        self.pool = None


if __name__ == "__main__":
    app = AlignmentTool(sys.argv)
    sys.exit(app.exec())
