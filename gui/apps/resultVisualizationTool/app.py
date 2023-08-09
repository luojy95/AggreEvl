import sys

import yaml

sys.path.append(".")
from typing import Optional
from PySide6 import QtCore, QtWidgets, QtGui

import os
import matplotlib
import glob

matplotlib.use("Qt5Agg")
from open3d.visualization import gui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import ticker
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

from PIL import Image
from gui.test.lazypixmap import test_linescan
from gui.test.pointcloud import PointCloudViewer
from gui.widgets.colorbarLabel import ColorBarLabel
from gui.widgets.lazyCanvas.gocatorFrame import GocatorFrame


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.tight_layout()
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

        self.cur_upperbound = None
        self.cur_lowerbound = None
        self.cur_avg = None

    def drawGradation(self, passing_proportion):
        self.axes.clear()

        passing_levels = np.array([3 / 8, 1 / 2, 3 / 4, 1, 1.5, 2, 2.5, 3])
        passing_levels_mm = passing_levels * 25.4

        self.axes.plot(
            passing_levels_mm,
            np.array(self._size_filter(passing_proportion)) * 100,
            "-",
            color="#FF5F05",
        )
        # self.axes.set_title("Gradation Curve")
        self.axes.set_xlabel("Grain size (mm)")
        self.axes.set_xscale("log", base=10)
        self.axes.tick_params(axis="both", which="major")
        self.axes.tick_params(axis="both", which="minor")
        self.axes.set_ylabel("Percentage passing (%)")
        self.axes.set_title("Ballast Size Distribution over 3/8 inch.")

        for axis in [self.axes.xaxis, self.axes.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_minor_formatter(ticker.ScalarFormatter())

        self.axes.grid(which="both", axis="both")

        self.draw()

    def drawGradationNew(
        self,
        passing_proportion,
        upperbound=None,
        lowerbound=None,
        current_passing_propotion=None,
    ):
        self.axes.clear()

        passing_levels = np.array([3 / 8, 1 / 2, 3 / 4, 1, 1.5, 2, 2.5, 3])
        passing_levels_mm = passing_levels * 25.4

        curves = []
        labels = []
        if passing_proportion is not None:
            avg = self.axes.plot(
                passing_levels_mm,
                np.array(passing_proportion) * 100,
                "-",
                color="#FF5F05",
                label="Average PSD",
            )[0]
            curves.append(avg)
            labels.append("Average PSD")
            self.cur_avg = passing_proportion
        if current_passing_propotion is not None:
            cur = self.axes.plot(
                passing_levels_mm,
                np.array(current_passing_propotion) * 100,
                "-",
                color="#009FD4",
                label="Current Analysis Window PSD",
            )[0]
            curves.append(cur)
            labels.append("Current Analysis Window PSD")
        if upperbound is not None and lowerbound is not None:
            error_band = self.axes.fill_between(
                passing_levels_mm,
                lowerbound * 100,
                upperbound * 100,
                color="#FF5F05",
                alpha=0.2,
                label="PSD Range",
            )
            curves.append(error_band)
            labels.append("PSD Range")
            self.cur_lowerbound = lowerbound
            self.cur_upperbound = upperbound
        self.axes.set_xlabel("Grain size (mm)")
        self.axes.set_xscale("log", base=10)
        self.axes.tick_params(axis="both", which="major")
        self.axes.tick_params(axis="both", which="minor")
        self.axes.set_ylabel("Percentage passing (%)")
        self.axes.set_title("Particle Size Distribution over 3/8 inch.")

        for axis in [self.axes.xaxis, self.axes.yaxis]:
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_minor_formatter(ticker.ScalarFormatter())

        self.axes.grid(which="both", axis="both")
        self.axes.legend(handles=tuple(curves), labels=tuple(labels), loc=2)
        self.draw()

    def saveFig(self, path):
        self.fig.savefig(path)

    def update_current_psd(self, cur_passing_proportion):
        self.drawGradationNew(
            self.cur_avg,
            self.cur_upperbound,
            self.cur_lowerbound,
            cur_passing_proportion,
        )

    def _size_filter(self, l):
        total = l[-1] - l[0]
        ret = []
        for i in range(len(l) - 1):
            ret.append((l[i + 1] - l[0]) / total)
        ret.append(1)
        return ret


from gui.resourcepool import ResourcePool
from tools.logger import default_logger as logger
from tools.imagesizeutil import get_image_size

from gui.widgets.lazyCanvas.lazyCanvas import LazySequentialCanvas, LazyCanvas
from gui.widgets.lazyCanvas.lazyGraphicsItem import LazyGraphicsPixmapItem
from gui.widgets.lazyCanvas.sequentialPIxmapOverlayCanvas import (
    SequentialPixmapOverlayCanvas,
)
from gui.widgets.lazyCanvas.sequentialPixmapOverlayCanvasV2 import (
    SequentialPixmapOverlayCanvasV2,
)

import cv2
import pandas as pd
import numpy as np
import tqdm

from bisect import bisect_left
from datetime import datetime, timedelta
from gui.worker import Worker


class ResultVisualizationWidget(QtWidgets.QWidget):
    MINIMUM_HEIGHT_CANVAS = 1280
    MINIMUM_WIDTH_CANVAS = 360

    LABEL_OVERLAY_INDEX = 0
    GOCATOR_OVERLAY_INDEX = 1

    def __init__(self, resourcepool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        main_layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QHBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout, 4)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

        titleFont = QtGui.QFont()
        titleFont.setBold(True)

        seglayout = QtWidgets.QVBoxLayout()
        segmentated_canvas_label = QtWidgets.QLabel("Segmentation Overlay View")
        segmentated_canvas_label.setFont(titleFont)
        segmentated_canvas_label.setMargin(10)

        self._segmented_canvas = LazySequentialCanvas(resourcepool)
        self._splitted_canvas = LazySequentialCanvas(resourcepool)
        self._gocator_canvas = LazySequentialCanvas(resourcepool)
        self.segmented_overlay_canvas = SequentialPixmapOverlayCanvasV2(resourcepool)
        self.gradation_canvas = LazySequentialCanvas(resourcepool)
        self.fi_canvas = LazyCanvas(resourcepool)
        self.gps_canvas = LazyCanvas(resourcepool)
        self.segmented_overlay_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.segmented_overlay_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS)
        self.segmented_overlay_canvas.view.is_zoom_by_scroll_enabled = False
        self.segmented_overlay_canvas.view.is_hover_show_analysis_rect_enabled = True
        self.segmented_overlay_canvas.view.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.segmented_overlay_canvas.view.verticalScrollBar().valueChanged.connect(
            self._syncScrollbar
        )
        seglayout.addWidget(segmentated_canvas_label)
        seglayout.addWidget(self.segmented_overlay_canvas)

        filayout = QtWidgets.QVBoxLayout()
        fi_canvas_label = QtWidgets.QLabel("Fouling Index")
        fi_canvas_label.setFont(titleFont)
        fi_canvas_label.setMargin(10)
        self.fi_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.fi_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS // 4)
        self.fi_canvas.view.verticalScrollBar().valueChanged.connect(
            self._syncScrollbar
        )
        self.fi_canvas.view.is_zoom_by_scroll_enabled = False
        self.fi_canvas.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        filayout.addWidget(fi_canvas_label)
        filayout.addWidget(self.fi_canvas)

        gradation_dist_layout = QtWidgets.QVBoxLayout()
        gradation_canvas_label = QtWidgets.QLabel("Particle Size Distribution")
        gradation_canvas_label.setFont(titleFont)
        gradation_canvas_label.setMargin(10)
        self.gradation_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.gradation_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS // 4)
        self.gradation_canvas.view.verticalScrollBar().valueChanged.connect(
            self._syncScrollbar
        )
        self.gradation_canvas.view.is_zoom_by_scroll_enabled = False
        self.gradation_canvas.view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        gradation_dist_layout.addWidget(gradation_canvas_label)
        gradation_dist_layout.addWidget(self.gradation_canvas)

        gpslayout = QtWidgets.QVBoxLayout()
        gps_canvas_label = QtWidgets.QLabel("GPS")
        gps_canvas_label.setFont(titleFont)
        gps_canvas_label.setMargin(10)

        self.gps_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.gps_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS // 4)
        self.gps_canvas.view.verticalScrollBar().valueChanged.connect(
            self._syncScrollbar
        )
        self.gps_canvas.view.is_zoom_by_scroll_enabled = False

        self.gps_canvas.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        gpslayout.addWidget(gps_canvas_label)
        gpslayout.addWidget(self.gps_canvas)

        left_layout.addLayout(seglayout, 4)
        left_layout.addLayout(gradation_dist_layout, 2)
        left_layout.addLayout(filayout, 1)
        left_layout.addLayout(gpslayout, 1)

        gradationlayout = QtWidgets.QVBoxLayout()
        gradationlabel = QtWidgets.QLabel("Particle Size Distribution (PSD) Curve")
        gradationlabel.setFont(titleFont)
        self.gradation_plot = MplCanvas()

        gradationlayout.addWidget(gradationlabel)
        gradationlayout.addWidget(self.gradation_plot)
        gradationlayout.setContentsMargins(10, 0, 10, 10)

        segment_overlay_alpha_label = QtWidgets.QLabel("2D Segmentation Label Opacity")
        segment_overlay_alpha_label.setFont(titleFont)
        slider_layout = QtWidgets.QHBoxLayout()
        self.segment_overlay_alpha_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.segment_overlay_alpha_slider.setValue(
            SequentialPixmapOverlayCanvasV2.DEFAULT_ALPHA * 100
        )
        self.segment_overlay_alpha_slider.valueChanged.connect(
            self._adjust_overlay_alpha
        )
        self.segment_overlay_alpha_text = QtWidgets.QLabel(
            f"{SequentialPixmapOverlayCanvasV2.DEFAULT_ALPHA:.2f}"
        )
        slider_layout.setContentsMargins(10, 10, 10, 0)
        slider_layout.addWidget(self.segment_overlay_alpha_slider)
        slider_layout.addWidget(self.segment_overlay_alpha_text)
        segment_overlay_alpha_label.setMargin(10)

        gocator_overlay_alpha_label = QtWidgets.QLabel("3D Scanner Heightmap Opacity")
        gocator_overlay_alpha_label.setFont(titleFont)
        gocator_slider_layout = QtWidgets.QHBoxLayout()
        self.gocator_overlay_alpha_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.gocator_overlay_alpha_slider.setValue(
            SequentialPixmapOverlayCanvasV2.DEFAULT_ALPHA * 100
        )
        self.gocator_overlay_alpha_slider.valueChanged.connect(
            self._adjust_overlay_alpha_gocator
        )
        self.gocator_overlay_alpha_text = QtWidgets.QLabel(
            f"{SequentialPixmapOverlayCanvasV2.DEFAULT_ALPHA:.2f}"
        )
        gocator_slider_layout.setContentsMargins(10, 10, 10, 0)
        gocator_slider_layout.addWidget(self.gocator_overlay_alpha_slider)
        gocator_slider_layout.addWidget(self.gocator_overlay_alpha_text)
        gocator_overlay_alpha_label.setMargin(10)

        segment_overlay_scale_label = QtWidgets.QLabel("2D Segmentation View Scale")
        segment_overlay_scale_label.setFont(titleFont)
        scale_slider_layout = QtWidgets.QHBoxLayout()
        self.segment_overlay_scale_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.segment_overlay_scale_slider.setValue(25)
        self.segment_overlay_scale_slider.valueChanged.connect(
            self._adjust_overlay_scale
        )
        self.segment_overlay_scale_text = QtWidgets.QLabel(f"{1.0:.2f}")
        scale_slider_layout.setContentsMargins(10, 10, 10, 0)
        scale_slider_layout.addWidget(self.segment_overlay_scale_slider)
        scale_slider_layout.addWidget(self.segment_overlay_scale_text)
        segment_overlay_scale_label.setMargin(10)

        gocatorcolorbarlayout = QtWidgets.QVBoxLayout()
        gocatorcolorbar_label = QtWidgets.QLabel("3D Scanner Height (mm)")
        gocatorcolorbar_label.setFont(titleFont)
        self.gocatorcolorbar = ColorBarLabel(colors=GocatorFrame.colors)
        gocatorcolorbarlayout.addWidget(gocatorcolorbar_label)
        gocatorcolorbarlayout.addWidget(self.gocatorcolorbar)
        gocatorcolorbarlayout.setContentsMargins(10, 30, 10, 0)

        psdcolorbarlayout = QtWidgets.QVBoxLayout()
        psdcolorbar_label = QtWidgets.QLabel("Particle Size Distribution (PSD) (in.)")
        psdcolorbar_label.setFont(titleFont)
        self.psdcolorbar1 = ColorBarLabel(
            colors=[
                "#f72585",
                "#b5179e",
                "#7209b7",
                "#3a0ca3",
            ]
        )
        self.psdcolorbar1.values = [
            "3/8 < size <= 1/2",
            "1/2 < size <= 3/4",
            "3/4 < size <= 1.0",
            "1.0 < size <= 1.5",
        ]
        self.psdcolorbar2 = ColorBarLabel(
            colors=[
                "#4361ee",
                "#4895ef",
                "#4cc9f0",
            ]
        )
        self.psdcolorbar2.values = [
            "1.5 < size <= 2.0",
            "2.0 < size <= 2.5",
            "2.5 < size <= 3.0",
        ]

        psdcolorbarlayout.addWidget(psdcolorbar_label)
        psdcolorbarlayout.addWidget(self.psdcolorbar1)
        psdcolorbarlayout.addWidget(self.psdcolorbar2)
        psdcolorbarlayout.setContentsMargins(10, 10, 10, 10)

        colorbarlayout = QtWidgets.QVBoxLayout()
        colorbar_label = QtWidgets.QLabel("Fouling Index (FI) Legend")
        colorbar_label.setFont(titleFont)
        self.colorbar = ColorBarLabel(
            colors=["#00b800", "#ffff00", "#ff7f00", "#ff0000", "#0000ff"]
        )
        self.colorbar.values = [
            " 0 <= FI < 10",
            "10 <= FI < 20",
            "20 <= FI < 30",
            "30 <= FI < 40",
            "40 <= FI < 50",
        ]
        colorbarlayout.addWidget(colorbar_label)
        colorbarlayout.addWidget(self.colorbar)
        colorbarlayout.setContentsMargins(10, 10, 10, 10)

        quantified_metrics_layout = QtWidgets.QVBoxLayout()

        quantified_result_label = QtWidgets.QLabel("Quantified Metrics")
        quantified_result_label.setFont(titleFont)

        metrics_font = QtGui.QFont()
        metrics_font.setPointSize(13)
        metrics_layout = QtWidgets.QHBoxLayout()
        metrics_layout.setContentsMargins(20, 0, 20, 0)
        self.num_particle_label = QtWidgets.QLabel("0 Particle(s) Detected: ")
        self.num_particle_label.setMargin(20)
        self.num_particle_label.setFont(metrics_font)
        self.pds_label = QtWidgets.QLabel("Average PDS = ")
        self.pds_label.setMargin(20)
        self.pds_label.setFont(metrics_font)
        self.fi_label = QtWidgets.QLabel("Average FI = ")
        self.fi_label.setMargin(20)
        self.fi_label.setFont(metrics_font)

        metrics_layout.addWidget(self.num_particle_label)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self.pds_label)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self.fi_label)

        quantified_metrics_layout.addWidget(quantified_result_label)
        quantified_metrics_layout.addLayout(metrics_layout)
        quantified_metrics_layout.setContentsMargins(10, 40, 10, 40)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(10, 40, 10, 10)
        self.visualize_point_cloud_btn = QtWidgets.QPushButton(
            "Visualize Point Cloud(s)"
        )
        self.export_btn = QtWidgets.QPushButton("Generate Final Report")
        self.export_btn.setStyleSheet(
            "QPushButton {background-color:green; color:white;}"
        )
        self.export_btn.clicked.connect(self._generate_final_report)
        btn_layout.addStretch()
        btn_layout.addWidget(self.visualize_point_cloud_btn)
        self.visualize_point_cloud_btn.clicked.connect(
            self._open_point_cloud_visualizer
        )
        btn_layout.addWidget(self.export_btn)

        right_layout.addLayout(gocatorcolorbarlayout)
        right_layout.addLayout(psdcolorbarlayout)
        right_layout.addLayout(colorbarlayout)
        right_layout.addLayout(gradationlayout)
        right_layout.addWidget(segment_overlay_scale_label)
        right_layout.addLayout(scale_slider_layout)
        right_layout.addWidget(segment_overlay_alpha_label)
        right_layout.addLayout(slider_layout)
        right_layout.addWidget(gocator_overlay_alpha_label)
        right_layout.addLayout(gocator_slider_layout)
        right_layout.addLayout(quantified_metrics_layout)
        right_layout.addStretch()
        right_layout.addLayout(btn_layout)
        self.colorbar.setMinimumWidth(640)

        self.times = None
        self.gps_querybook = None
        self.gps_map = None
        self.gps_info = None

        self.current_scale = 1.0

        self.point_cloud_root = None
        self.threadpool = QtCore.QThreadPool(self)
        self.output_root = None

        self.gui_initialized = False

        self.gradations = None

        self.plot_worker = None

        self.scene_rate = 1.0

    def _generate_final_report(self):
        if self.output_root is None:
            return
        self.gradation_plot.update_current_psd(None)
        self.gradation_plot.saveFig(os.path.join(self.output_root, "gradation.png"))

        """Heatmap ColorBar"""
        fig, ax = plt.subplots(figsize=(9.25, 13))
        colors = ["#00b800", "#ffff00", "#ff7f00", "#ff0000", "#0000ff"]
        bounds = [0, 10, 20, 30, 40, 50]

        patches = []
        colors = colors[::-1]
        bounds = bounds[::-1]
        for i in range(len(colors)):
            patches.append(
                mpatches.Patch(
                    facecolor=colors[i],
                    label=f"{round(bounds[i+1]):d}<=FI<={round(bounds[i]):d}",
                    linewidth=1,
                    edgecolor="black",
                ),
            )
        plt.axis("equal")
        plt.axis("off")
        legend = fig.legend(
            handles=patches,
            prop={"size": 60},
            title="Fouling Index",
            fontsize=60,
            frameon=False,
        )
        plt.setp(legend.get_title(), fontsize=60)
        fig.savefig(os.path.join(self.output_root, "fi_legend.png"))
        fig.clear()

        """3D Scanner ColorBar"""
        fig, ax = plt.subplots(figsize=(6, 13))
        colors = GocatorFrame.colors
        bounds = self.gocatorcolorbar.values

        patches = []
        colors = colors[::-1]
        bounds = bounds[::-1]
        for color, val in zip(colors, bounds):
            patches.append(
                mpatches.Patch(
                    facecolor=color,
                    label=f"{round(val):d}",
                    linewidth=1,
                    edgecolor="black",
                ),
            )
        plt.axis("equal")
        plt.axis("off")
        legend = fig.legend(
            handles=patches,
            prop={"size": 60},
            title="Height (mm)",
            fontsize=60,
            frameon=False,
        )
        plt.setp(legend.get_title(), fontsize=60)
        fig.savefig(os.path.join(self.output_root, "gocator_legend.png"))
        fig.clear()

        worker = Worker(
            self._export_all_data, progress_callback=self._progress_callback
        )
        worker.signals.finished.connect(self._generate_complete)

        self.threadpool.start(worker)

    def _progress_callback(self, value):
        pass

    def _generate_complete(self):
        QtWidgets.QMessageBox.information(
            self,
            "Congrats!",
            f"Successfully generated final report to {os.path.relpath(self.output_root)}/final_report.png!",
        )

    def _export_all_data(self, progress_callback=None):
        logger.info("[ResultVisualizer]Export all data")
        if (
            self.output_root is not None
            and os.path.exists(self.output_root)
            and os.path.isdir(self.output_root)
        ):
            output_seq_width = 1024

            path = self.output_root

            linescan_results_path = os.path.join(path, "linescan")
            """Splitted"""
            splitted_path = os.path.join(linescan_results_path, "splitted")
            y = 0
            group = []
            rate = 1.0
            for file in sorted(glob.glob(os.path.join(splitted_path, "*.png"))):
                target_width = output_seq_width
                width, height = get_image_size(file)
                target_height = round(target_width / width * height)
                rate = target_width / width
                y += target_height
                group.append(
                    np.asarray(Image.open(file).resize((target_width, target_height)))
                )
            combined = np.concatenate(group, axis=0)
            print(f"output_splitted_height: {y}")
            """Segmented"""
            segmented_path = os.path.join(linescan_results_path, "segmented")
            y = 0
            group = []
            for file in sorted(glob.glob(os.path.join(segmented_path, "*.png"))):
                target_width = output_seq_width
                width, height = get_image_size(file)
                target_height = round(target_width / width * height)
                y += target_height
                group.append(
                    np.asarray(Image.open(file).resize((target_width, target_height)))
                )
            segmented_combined = np.concatenate(group, axis=0)
            print(f"output_segment_height: {y}")
            """3D Scanner"""
            gocator_results_path = os.path.join(path, "gocator")
            gocator_path = os.path.join(gocator_results_path, "smooth")
            gocator_metadata_path = os.path.join(gocator_results_path, "metadata.yaml")
            with open(gocator_metadata_path, "r") as f:
                gocator_metadata = yaml.safe_load(f)

            y = 0
            gocator_group = []
            start = None
            end = None
            for file in sorted(glob.glob(os.path.join(gocator_path, "*.png"))):
                width, height = get_image_size(file)
                target_width = round(width * 4 * rate)
                target_height = round(height * 4 * rate)
                y += target_height
                image = np.asarray(
                    Image.open(file).resize((target_width, target_height))
                )
                if start is None or end is None:
                    im_np_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    colsums = im_np_gray.sum(axis=0)

                    for i in range(colsums.shape[0]):
                        if colsums[i] / target_height > 0 and start is None:
                            start = i

                        if (
                            colsums[i] / target_height < 0
                            and end is None
                            and start is not None
                        ):
                            end = i

                    if start is None:
                        start = 0
                    else:
                        start += 1

                    if end is None:
                        end = target_width

                gocator_group.append(image[:, start:end])
                target_width = end - start
            gocator_combined = np.concatenate(gocator_group, axis=0)
            print(f"output_gocator_height: {y}")

            max_height = min(gocator_combined.shape[0], combined.shape[0])
            combined_img = Image.fromarray(combined[:max_height])
            combined_img.save(os.path.join(path, "splitted_merged.png"))
            segmented_combined_img = Image.fromarray(segmented_combined[:max_height])
            segmented_combined_img.save(os.path.join(path, "segment_merged.png"))
            

            """Overlay"""
            translation_x = round(
                -gocator_metadata["linescan_align_offset"] * rate + start
            )

            overlay = segmented_combined[:max_height]
            alpha = 0.6
            if translation_x > 0:
                overlay_width = min(overlay.shape[1] - translation_x, target_width)
                overlay[:, translation_x : translation_x + overlay_width] = (
                    overlay[:, translation_x : translation_x + overlay_width] * (1 - alpha)
                    + gocator_combined[:max_height, :overlay_width] * alpha
                )
            else:
                overlay_width = min(overlay.shape[0], target_width + translation_x)
                overlay[:, :overlay_width] = (
                    overlay[:, :overlay_width] * (1 - alpha)
                    + gocator_combined[:max_height, -translation_x:overlay_width-translation_x] * alpha
                )
            overlayimg = Image.fromarray(overlay)
            overlayimg.save(os.path.join(path, "overlay_merged.png"))

            gocator_combined_img_np = gocator_combined[:max_height]

            gocator_combined_colsum = gocator_combined_img_np[:,:].sum(axis=2).sum(axis=0)
            start = None
            end = None
            for ci in range(gocator_combined_colsum.shape[0]):
                if gocator_combined_colsum[ci] != 0 and start is None:
                    start = ci 
                
                elif gocator_combined_colsum[ci] == 0 and start is not None:
                    end = ci 
                    break 
            

            gocator_combined_img = Image.fromarray(gocator_combined_img_np[:, start:end])
            gocator_combined_img.save(os.path.join(path, "gocator_merged.png"))

            print(f"output_overlay_height: {max_height}")

            """GPS"""
            gps_image = np.zeros_like(overlay) + 255  # initialize as white
            if self.gps_info is not None:  # verify if gps data is loaded
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Blue color in BGR
                color = (0, 0, 0)

                # Line thickness of 2 px
                thickness = 10

                # fontScale
                fontScale = 4
                padding = 50
                actual_height = max_height
                original_height = self.times[-1][0]
                ratio = actual_height / original_height
                for i in range(self.times.shape[0]):
                    if self.gps_info[i] != None:
                        actual_height_i = self.times[i][0] * ratio
                        lat, lng = self.gps_info[i]
                        lat_label = "N"
                        if lat == 0:
                            lat_label = ""
                        elif lat < 0:
                            lat_label = "S"

                        lng_label = "E"
                        if lng == 0 or lng == 180 or lng == -180:
                            lng_label = ""
                        elif lng < 0:
                            lng_label = "W"

                        lat = np.abs(lat)
                        lng = np.abs(lng)

                        s1 = f"{lat:.6f} " + lat_label
                        s2 = f"{lng:.6f} " + lng_label

                        size, _ = cv2.getTextSize(s1, font, fontScale, thickness)
                        text_width, text_height = size

                        # org
                        org = (50, round(actual_height_i) + text_height + padding)
                        if (
                            org[1] + 2 * text_height + padding > gps_image.shape[0]
                        ):  # don't draw incomplete gps
                            continue

                        # Using cv2.putText() method
                        gps_image = cv2.putText(
                            gps_image,
                            s1,
                            org,
                            font,
                            fontScale,
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        gps_image = cv2.putText(
                            gps_image,
                            s2,
                            (org[0], org[1] + padding * 2 + text_height),
                            font,
                            fontScale,
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )
            gpsimg = Image.fromarray(gps_image)
            gpsimg.save(os.path.join(path, "gps_merged.png"))

            """FI"""
            heatmap_path = os.path.join(
                linescan_results_path,
                "newheatmaps",
                "newheatmap_all_final.png",
            )
            heatmap_img = Image.open(heatmap_path)
            hsize = heatmap_img.size
            newsize = (
                round(overlayimg.size[1] / hsize[1] * hsize[0]),
                overlayimg.size[1],
            )
            heatmap_img = heatmap_img.resize(newsize)
            heatmap_img.save(os.path.join(path, "heatmap.png"))

            pds_text = self.pds_label.text()
            fi_text = self.fi_label.text()
            detection_text = self.num_particle_label.text()

            gocator_legend = Image.open(os.path.join(path, "gocator_legend.png"))
            fsize = gocator_legend.size
            newsize = (fsize[0] * 2, fsize[1] * 2)
            gocator_legend = gocator_legend.resize(newsize)

            fi_legend = Image.open(os.path.join(path, "fi_legend.png"))
            fsize = fi_legend.size
            newsize = (fsize[0] * 2, fsize[1] * 2)
            fi_legend = fi_legend.resize(newsize)

            print(fi_legend.size)

            gradation_img = Image.open(os.path.join(path, "gradation.png"))

            psd_img = Image.open(os.path.join(path, "linescan", "gradation_merged.png"))
            psd_img = psd_img.resize(overlayimg.size)

            psd_legend = Image.open(
                os.path.join(path, "linescan", "gradation_distribution_legend.png")
            )
            fsize = psd_legend.size
            newsize = (fsize[0] * 2, fsize[1] * 2)
            psd_legend = psd_legend.resize(newsize)
            print(psd_legend.size)

            """Final Composition"""

            padding_width = 200
            padding_height = 100
            title_height = 800
            subtitle_height = 400
            footnote_height = 1600
            if combined_img.height >= fi_legend.height + psd_legend.height:
                total_width = (
                    padding_width
                    + combined_img.width
                    + padding_width
                    + segmented_combined_img.width
                    + padding_width
                    + gocator_combined_img.width
                    + padding_width
                    + gocator_legend.width
                    + padding_width
                    + overlayimg.width
                    + padding_width
                    + psd_img.width
                    + padding_width
                    + heatmap_img.width
                    + padding_width
                    + psd_legend.width
                    + padding_width
                    + gpsimg.width
                    + padding_width
                )
            else:
                total_width = (
                    padding_width
                    + combined_img.width
                    + padding_width
                    + segmented_combined_img.width
                    + padding_width
                    + gocator_combined_img.width
                    + padding_width
                    + gocator_legend.width
                    + padding_width
                    + overlayimg.width
                    + padding_width
                    + psd_img.width
                    + padding_width
                    + psd_legend.width
                    + padding_width
                    + heatmap_img.width
                    + padding_width
                    + fi_legend.width
                    + padding_width
                    + gpsimg.width
                    + padding_width
                )
            total_height = (
                title_height
                + padding_height
                + combined_img.height
                + padding_height
                + subtitle_height
                + padding_height
                + footnote_height
                + padding_height
                + subtitle_height
                + padding_height
            )

            report = np.zeros((total_height, total_width, 3)).astype(np.uint8) + 255

            """Title"""
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Blue color in BGR
            color = (0, 0, 0)

            # Line thickness of 2 px
            thickness = 2

            # fontScale
            fontScale = 1
            padding = 5

            title = "Ballast Fouling Condition Analysis Report"
            titleFontScale = 10
            titleThinkness = 20
            size, _ = cv2.getTextSize(title, font, titleFontScale, titleThinkness)
            text_width, text_height = size

            print(text_width, text_height)

            text_org_width = (total_width - text_width) // 2
            text_org_height = (title_height - text_height) // 2 + text_height

            # org
            org = (text_org_width, text_org_height)
            report = cv2.putText(
                report,
                title,
                org,
                font,
                titleFontScale,
                color,
                titleThinkness,
                cv2.LINE_AA,
            )

            subtitle = "subtitle"
            subtitleFontScale = 6
            subtitleThinkness = 10
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            _, subtitle_text_height = size
            subtitle_org_height = (
                title_height
                + padding_height
                + combined_img.height
                + padding_height
                + (subtitle_height + subtitle_text_height) // 2
            )

            """Canvas"""
            canvas_start = title_height + padding_height
            cur_start = padding_width
            report[
                canvas_start : canvas_start + gocator_legend.height,
                cur_start : cur_start + gocator_legend.width,
            ] = np.asarray(gocator_legend)[:, :, :3]

            cur_start += gocator_legend.width + padding_width
            report[
                canvas_start : canvas_start + gocator_combined_img.height,
                cur_start : cur_start + gocator_combined_img.width,
            ] = np.asarray(gocator_combined_img)[:, :, :3]
            subtitle = "(a)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = (
                cur_start + (gocator_combined_img.width - subtitle_text_width) // 2
            )
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += gocator_combined_img.width + padding_width
            report[
                canvas_start : canvas_start + gpsimg.height,
                cur_start : cur_start + gpsimg.width,
            ] = np.asarray(gpsimg)[:, :, :3]
            subtitle = "(b)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = cur_start + (gpsimg.width - subtitle_text_width) // 2
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += gpsimg.width + padding_width
            report[
                canvas_start : canvas_start + combined_img.height,
                cur_start : cur_start + combined_img.width,
            ] = np.asarray(combined_img)[:, :, :3]
            subtitle = "(c)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = (
                cur_start + (combined_img.width - subtitle_text_width) // 2
            )
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += combined_img.width + padding_width
            report[
                canvas_start : canvas_start + segmented_combined_img.height,
                cur_start : cur_start + segmented_combined_img.width,
            ] = np.asarray(segmented_combined_img)[:, :, :3]
            subtitle = "(d)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = (
                cur_start + (segmented_combined_img.width - subtitle_text_width) // 2
            )
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += segmented_combined_img.width + padding_width
            report[
                canvas_start : canvas_start + overlayimg.height,
                cur_start : cur_start + overlayimg.width,
            ] = np.asarray(overlayimg)[:, :, :3]
            subtitle = "(e)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = (
                cur_start + (overlayimg.width - subtitle_text_width) // 2
            )
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += overlayimg.width + padding_width
            report[
                canvas_start : canvas_start + psd_img.height,
                cur_start : cur_start + psd_img.width,
            ] = np.asarray(psd_img)[:, :, :3]
            subtitle = "(f)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = cur_start + (psd_img.width - subtitle_text_width) // 2
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += psd_img.width + padding_width
            if combined_img.height < fi_legend.height + psd_legend.height:
                report[
                    canvas_start : canvas_start + psd_legend.height,
                    cur_start : cur_start + psd_legend.width,
                ] = np.asarray(psd_legend)[:, :, :3]

                cur_start += psd_legend.width + padding_width

            report[
                canvas_start : canvas_start + heatmap_img.height,
                cur_start : cur_start + heatmap_img.width,
            ] = np.asarray(heatmap_img)[:, :, :3]
            subtitle = "(g)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = (
                cur_start + (heatmap_img.width - subtitle_text_width) // 2
            )
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                org,
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            cur_start += heatmap_img.width + padding_width
            if combined_img.height >= fi_legend.height + psd_legend.height:
                report[
                    canvas_start : canvas_start + psd_legend.height,
                    cur_start : cur_start + psd_legend.width,
                ] = np.asarray(psd_legend)[:, :, :3]
                report[
                    canvas_start
                    + psd_legend.height : canvas_start
                    + fi_legend.height
                    + psd_legend.height,
                    cur_start : cur_start + fi_legend.width,
                ] = np.asarray(fi_legend)[:, :, :3]
                cur_start += psd_legend.width + padding_width
            else:
                report[
                    canvas_start : canvas_start + fi_legend.height,
                    cur_start : cur_start + fi_legend.width,
                ] = np.asarray(fi_legend)[:, :, :3]
                cur_start += fi_legend.width + padding_width

            footnote_start = (
                title_height
                + padding_height
                + combined_img.height
                + padding_height
                + subtitle_height
                + padding_height
            )
            width = round(
                gradation_img.size[0] * footnote_height / gradation_img.size[1]
            )
            gradation_img = gradation_img.resize((width, footnote_height))
            gradation_img_start = padding_width + gocator_legend.width + padding
            report[
                footnote_start
                - padding_height // 2 : footnote_start
                - padding_height // 2
                + gradation_img.height,
                gradation_img_start : gradation_img_start + gradation_img.width,
            ] = np.asarray(gradation_img)[:, :, :3]
            subtitle = "(h)"
            size, _ = cv2.getTextSize(
                subtitle, font, subtitleFontScale, subtitleThinkness
            )
            subtitle_text_width, subtitle_text_height = size
            subtitle_org_width = (
                gradation_img_start + (gradation_img.width - subtitle_text_width) // 2
            )
            subtitle_org_height = (
                footnote_start
                - padding_height // 2
                + gradation_img.height
                + padding_height
                + subtitle_text_height
            )
            org = (subtitle_org_width, subtitle_org_height)
            report = cv2.putText(
                report,
                subtitle,
                (subtitle_org_width, subtitle_org_height),
                font,
                subtitleFontScale,
                color,
                subtitleThinkness,
                cv2.LINE_AA,
            )

            footnote = "footnote"
            footnoteFontScale = 4
            footnoteThinkness = 10
            size, _ = cv2.getTextSize(
                footnote, font, footnoteFontScale, footnoteThinkness
            )
            _, footnote_text_height = size

            footnotes = [
                detection_text + " " + fi_text + ", " + pds_text,
                "(a) 3D Scanner Heightmap",
                "(b) GPS Coordinates",
                "(c) Line Scan Camera Image",
                "(d) 2D Segmentated Image",
                "(e) Overlay of 3D Scanner Heightmap and 2D Segmented Image",
                "(f) Fouling Index Distribution",
                "(g) Particle Size Distribution along Railroad over 3/8 inch.",
                "(h) Particle Size Distribution over 3/8 inch.",
            ]
            footnote_padding = 100
            footnote_org_height = footnote_start + footnote_padding
            footnote_org_width = (
                padding_width + gocator_legend.width + padding + width + padding_width
            )
            for i in range(len(footnotes)):
                org = (
                    footnote_org_width,
                    footnote_org_height
                    + i * footnote_padding
                    + i * footnote_text_height,
                )
                report = cv2.putText(
                    report,
                    footnotes[i],
                    org,
                    font,
                    footnoteFontScale,
                    color,
                    footnoteThinkness if i > 0 else 15,
                    cv2.LINE_AA,
                )

            report_img = Image.fromarray(report)

            report_img.save(os.path.join(path, "final_report.png"))

    def _open_point_cloud_visualizer(self):
        worker = Worker(self._visualizer_internal)
        self.threadpool.start(worker)

    def _visualizer_internal(self, progress_callback=None):
        if (
            self.point_cloud_root is not None
            and os.path.exists(self.point_cloud_root)
            and os.path.isdir(self.point_cloud_root)
        ):
            # gui.Application.instance.initialize()
            # PointCloudViewer(os.path.relpath(self.point_cloud_root))
            # gui.Application.instance.run()
            cmd = '/bin/bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate i-ballast && '
            cmd += "cd /home/kelin/Documents/BSVDataKit && "
            cmd += f'python3 gui/test/pointcloud.py --path {os.path.relpath(self.point_cloud_root)}"'
            os.system(cmd)

    def _syncScrollbar(self, value):
        self.segmented_overlay_canvas.view.verticalScrollBar().setValue(value)
        self.fi_canvas.view.verticalScrollBar().setValue(value)
        self.gradation_canvas.view.verticalScrollBar().setValue(value)
        self.gps_canvas.view.verticalScrollBar().setValue(value)
        if self.gradations is not None and self.plot_worker is None:
            if self.segmented_overlay_canvas.view.analysis_rect is not None:
                self.segmented_overlay_canvas.view.update_analysis_rect()

    def _create_plot_updator(self, center_y):
        if center_y is None:
            self.gradation_plot.update_current_psd(None)
        elif self.plot_worker is None:  # idle
            self.plot_worker = Worker(self._update_curve, center_y)
            self.plot_worker.signals.finished.connect(self._remove_plot_worker)
            self.threadpool.start(self.plot_worker)

    def _remove_plot_worker(self):
        self.plot_worker = None

    def _update_curve(self, center_y, progress_callback):
        if center_y is None:
            self.gradation_plot.update_current_psd(None)

        elif self.gradation_plot is not None and self.gradations is not None:
            row = round(center_y / self.scene_rate)
            if row < 0 or row >= self.gradations.shape[0]:
                self.gradation_plot.update_current_psd(None)
            else:
                current_passing_proportion = self.gradations[row]
                self.gradation_plot.update_current_psd(current_passing_proportion)

    def load_gps_from_cvs(self, csv_path):
        if not os.path.exists(csv_path):
            return

        gps_data = pd.read_csv(csv_path)

        print(gps_data.columns)

        lats = gps_data["ESRIGNSS_LATITUDE"]
        lngs = gps_data["ESRIGNSS_LONGITUDE"]
        ts = []
        for time_str in gps_data["ESRIGNSS_FIXDATETIME"]:
            t = datetime.strptime(time_str, "%m/%d/%Y %H:%M:%S")
            t = t.replace(microsecond=0)
            ts.append(t)
        self.gps_querybook = {"lats": lats, "lngs": lngs, "ts": ts}

        if self.times is not None:
            for j, t in enumerate(self.gps_querybook["ts"]):
                i = bisect_left(self.times[:, 1], t)

                if i >= self.times.shape[0]:
                    i = self.times.shape[0] - 1
                    d = t - self.times[i, 1]

                elif i == 0:
                    i = 0
                    d = self.times[i, 1] - t

                else:
                    if t - self.times[i - 1, 1] < self.times[i, 1] - t:
                        i = i - 1
                        d = t - self.times[i - 1, 1]
                    else:
                        d = self.times[i, 1] - t

                print(i, j, t, d)
                if self.gps_map[i] != None:
                    self.gps_map[i] = (
                        (j, d) if d < self.gps_map[i][1] else self.gps_map[i]
                    )
                else:
                    self.gps_map[i] = (j, d)
                self.gps_info[i] = (lats[self.gps_map[i][0]], lngs[self.gps_map[i][0]])
            print(self.gps_info)
            self.drawGps()

    def drawGps(self):
        if self.gps_info is None:
            return
        actual_height = self.getSegmentOverlayHeight()
        self.gps_canvas.clearSceneItems()
        self.gps_canvas.view.setSceneRect(
            0, 0, self.MINIMUM_WIDTH_CANVAS / 4, actual_height
        )
        original_height = self.times[-1][0]
        ratio = actual_height / original_height
        for i in range(self.times.shape[0]):
            if self.gps_info[i] != None:
                actual_height = self.times[i][0] * ratio
                lat, lng = self.gps_info[i]
                lat_label = "N"
                if lat == 0:
                    lat_label = ""
                elif lat < 0:
                    lat_label = "S"

                lng_label = "E"
                if lng == 0 or lng == 180 or lng == -180:
                    lng_label = ""
                elif lng < 0:
                    lng_label = "W"

                lat = np.abs(lat)
                lng = np.abs(lng)

                s = (
                    f"{lat:.6f}"
                    + "\N{DEGREE SIGN}"
                    + lat_label
                    + " \n "
                    + f"{lng:.6f}"
                    + "\N{DEGREE SIGN}"
                    + lng_label
                )

                item = QtWidgets.QGraphicsTextItem(s)
                item.setY(actual_height)
                transform = QtGui.QTransform()
                transform.scale(1.0, 1.0 / self.current_scale)
                item.setTransform(transform)
                item.setDefaultTextColor("white")
                self.gps_canvas.addItem(item)

    def scaleGps(self):
        for item in self.gps_canvas.scene.items():
            transform = QtGui.QTransform()
            transform.scale(1.0, 1.0 / self.current_scale)
            item.setTransform(transform)

    def _adjust_overlay_alpha(self, value):
        self.segment_overlay_alpha_text.setText(f"{value / 100.0:.2f}")
        return self.segmented_overlay_canvas.adjustForeGroundAlpha(
            self.LABEL_OVERLAY_INDEX, value / 100.0
        )

    def _adjust_overlay_alpha_gocator(self, value):
        self.gocator_overlay_alpha_text.setText(f"{value / 100.0:.2f}")
        return self.segmented_overlay_canvas.adjustForeGroundAlpha(
            self.GOCATOR_OVERLAY_INDEX, value / 100.0
        )

    def _adjust_overlay_scale(self, value):
        actual_scale = value / 100.0 * 2 + 0.5
        self.segment_overlay_scale_text.setText(f"{actual_scale:.1f}")
        # print(actual_scale)
        self.segmented_overlay_canvas.view.scale(
            1.0 / self.current_scale, 1.0 / self.current_scale
        )
        self.fi_canvas.view.scale(1.0, 1.0 / self.current_scale)
        self.gps_canvas.view.scale(1.0, 1.0 / self.current_scale)
        self.gradation_canvas.view.scale(1.0, 1.0 / self.current_scale)

        self.segmented_overlay_canvas.view.scale(actual_scale, actual_scale)
        self.fi_canvas.view.scale(1.0, actual_scale)
        self.gradation_canvas.view.scale(1.0, actual_scale)
        self.gps_canvas.view.scale(1.0, actual_scale)
        self.current_scale = actual_scale
        self.scaleGps()

    def loadResults(self, path):
        self.cleanup()
        self._syncScrollbar(0)
        self.segmented_overlay_canvas.view.setSceneRect(QtCore.QRectF())
        self.fi_canvas.view.setSceneRect(QtCore.QRectF())
        self.gps_canvas.view.setSceneRect(QtCore.QRectF())

        self.segment_overlay_alpha_slider.setValue(60)
        self.segment_overlay_scale_slider.setValue(25)
        logger.info("[ResultVisualizer]Load from path: " + path)
        if not os.path.exists(path) or not os.path.isdir(path):
            logger.error("[ResultVisualizer]Invalid output path")
        linescan_results_path = os.path.join(path, "linescan")

        self.output_root = path

        segmented_path = os.path.join(linescan_results_path, "segmented")
        y = 0
        for file in sorted(glob.glob(os.path.join(segmented_path, "*.png"))):
            target_width = self.MINIMUM_WIDTH_CANVAS
            width, height = get_image_size(file)
            target_height = round(target_width / width * height)
            item = LazyGraphicsPixmapItem(
                width=target_width,
                height=target_height,
                original_width=width,
                original_height=height,
            )
            item.setPath(file)
            item.setY(y)
            y += target_height
            self._segmented_canvas.addItem(item)
        self._segmented_canvas.view.initSceneRect()

        splitted_path = os.path.join(linescan_results_path, "splitted")
        y = 0
        rate = 1.0
        for file in sorted(glob.glob(os.path.join(splitted_path, "*.png"))):
            target_width = self.MINIMUM_WIDTH_CANVAS
            width, height = get_image_size(file)
            target_height = round(target_width / width * height)
            rate = target_width / width
            item = LazyGraphicsPixmapItem(
                width=target_width,
                height=target_height,
                original_width=width,
                original_height=height,
            )
            item.setPath(file)
            item.setY(y)
            y += target_height
            self._splitted_canvas.addItem(item)
        self._splitted_canvas.view.initSceneRect()
        print(f"splitted total height: {y}")
        self.segmented_overlay_canvas.view.analysis_rect_height = 2000 * rate
        self.segmented_overlay_canvas.view.signals.analysis_rect_updated.connect(
            self._create_plot_updator
        )
        self.scene_rate = rate

        gocator_results_path = os.path.join(path, "gocator")
        gocator_path = os.path.join(gocator_results_path, "smooth")

        gocator_metadata_path = os.path.join(gocator_results_path, "metadata.yaml")
        with open(gocator_metadata_path, "r") as f:
            gocator_metadata = yaml.safe_load(f)
            min_height = gocator_metadata["min_height_to_base"]
            max_height = gocator_metadata["max_height_to_base"]
            map_cvals = lambda cval: cval * (max_height - min_height) + min_height
            self.gocatorcolorbar.values = [
                map_cvals(cval) for cval in GocatorFrame.cvals
            ]
            self.gocatorcolorbar.update()
        y = 0
        for file in sorted(glob.glob(os.path.join(gocator_path, "*.png"))):
            width, height = get_image_size(file)
            target_width = round(width * 4 * rate)
            target_height = round(height * 4 * rate)
            y += target_height
            item = LazyGraphicsPixmapItem(
                width=target_width,
                height=target_height,
                original_width=width,
                original_height=height,
            )
            item.setPath(file)
            item.setY(y)
            item.setX(-gocator_metadata["linescan_align_offset"] * rate)
            self._gocator_canvas.addItem(item)
        print(f"gocator total height: {y}")

        self._gocator_canvas.view.initSceneRect()

        self.segmented_overlay_canvas.setForegroundCanvas(
            self._segmented_canvas, self.LABEL_OVERLAY_INDEX
        )
        self.segmented_overlay_canvas.setForegroundCanvas(
            self._gocator_canvas, self.GOCATOR_OVERLAY_INDEX
        )
        self.segmented_overlay_canvas.setBackgroundCanvas(self._splitted_canvas)

        self.segmented_overlay_canvas.adjustForeGroundAlpha(
            self.LABEL_OVERLAY_INDEX, self.segment_overlay_alpha_slider.value() / 100.0
        )
        self.segmented_overlay_canvas.requestRenderOverlay()

        print(self.getSegmentOverlayHeight())

        heatmap_path = os.path.join(
            linescan_results_path, "newheatmaps", "newheatmap_all_final.png"
        )

        if os.path.exists(heatmap_path):
            width, height = get_image_size(heatmap_path)
            target_height = self.getSegmentOverlayHeight()
            target_width = self.MINIMUM_WIDTH_CANVAS / 4

            heatmap_item = LazyGraphicsPixmapItem(
                width=target_width,
                height=target_height,
                original_width=width,
                original_height=height,
            )
            heatmap_item.setPath(heatmap_path)

            self.fi_canvas.addItem(heatmap_item)

        psd_path = os.path.join(linescan_results_path, "psd")

        if os.path.exists(psd_path):
            y = 0
            for file in sorted(glob.glob(os.path.join(psd_path, "*.png"))):
                target_width = self.MINIMUM_WIDTH_CANVAS / 2.5
                width, height = get_image_size(file)
                target_height = round(target_width / width * height) * 2.5
                item = LazyGraphicsPixmapItem(
                    width=target_width,
                    height=target_height,
                    original_width=width,
                    original_height=height,
                )
                item.setPath(file)
                item.setY(y)
                y += target_height
                self.gradation_canvas.addItem(item)
            print(f"psd total height: {y}")

        time_file = os.path.join(linescan_results_path, "time.txt")
        self.times = np.loadtxt(time_file).tolist()
        hr_diff_compare_with_GMT = 5  # IL
        for i in range(len(self.times)):
            m_time = self.times[i][1]
            t = datetime.fromtimestamp(m_time) + timedelta(
                hours=hr_diff_compare_with_GMT
            )
            t = t.replace(microsecond=0)
            self.times[i][1] = t
        self.times = np.array(self.times)
        self.gps_map = [None for i in range(self.times.shape[0])]
        self.gps_info = [None for i in range(self.times.shape[0])]

        segmentation_log_path = os.path.join(
            linescan_results_path, "2DSegmentationLog.txt"
        )
        if os.path.exists(segmentation_log_path):
            # legacy flow
            with open(segmentation_log_path, "r") as f:
                lines = f.readlines()
                passing_proportion = []
                pds = -1
                fi = -1
                for i in range(len(lines)):
                    if "Passing Proportion:" in lines[i]:
                        for s in lines[i].strip().split("[")[-1].split(" "):
                            try:
                                passing_proportion.append(float(s) * 100)
                            except:
                                pass
                        for s in lines[i + 1].strip().split("]")[0].split(" "):
                            try:
                                passing_proportion.append(float(s) * 100)
                            except:
                                pass
                    if "PDS:" in lines[i]:
                        # PDS: 18.77327606774786; FI: 6.217163922681153
                        pds = float(lines[i].strip().split(";")[0].split(" ")[-1])
                        fi = float(lines[i].strip().split(";")[-1].split(" ")[-1])

                print(passing_proportion, pds, fi)
                self.num_particle_label.setText("")
                self.pds_label.setText(f"Average PDS = {pds:.3f}")
                self.fi_label.setText(f"Average FI = {fi:.3f}")
                passing_proportion = np.array(passing_proportion)

                self.gradation_plot.drawGradation(passing_proportion)

        else:
            # new flow:
            pds_fi_path = os.path.join(linescan_results_path, "overall_metrics.txt")
            num_particles, pds, fi = np.loadtxt(pds_fi_path)
            self.num_particle_label.setText(
                f"{round(num_particles):d} Particle(s) Detected:"
            )
            self.pds_label.setText(f"Average PDS = {pds:.3f}")
            self.fi_label.setText(f"Average FI = {fi:.3f}")

            gradation_meta_path = os.path.join(
                linescan_results_path, "gradation_meta.txt"
            )
            gradation_metas = np.loadtxt(gradation_meta_path)

            gradations_path = os.path.join(
                linescan_results_path, "gradation_merged.txt"
            )

            current_passing_proportion = None
            if os.path.exists(gradations_path):
                self.gradations = np.loadtxt(gradations_path)

            upperbound = gradation_metas[0]
            lowerbound = gradation_metas[1]
            passing_proportion = gradation_metas[2]
            self.gradation_plot.drawGradationNew(
                passing_proportion, upperbound, lowerbound, current_passing_proportion
            )

        self.point_cloud_root = os.path.join(path, "pointclouds")

    def getSegmentOverlayHeight(self):
        if len(self.segmented_overlay_canvas.bgItemQueue) > 0:
            total_height = (
                self.segmented_overlay_canvas.bgItemQueue[-1].y()
                + self.segmented_overlay_canvas.bgItemQueue[-1].size.height()
            )
            return total_height
        return 0

    def cleanup(self):
        self.segmented_overlay_canvas.clearSceneItems()
        self._segmented_canvas.clearSceneItems()
        self._splitted_canvas.clearSceneItems()
        self._gocator_canvas.clearSceneItems()
        self.fi_canvas.clearSceneItems()


class ResultVisualizationMainWindow(QtWidgets.QMainWindow):
    def __init__(self, resourcepool, parent=None) -> None:
        super().__init__(parent)
        self.pool = resourcepool

        self.mainWidget = ResultVisualizationWidget(self.pool)
        self.setCentralWidget(self.mainWidget)

        self._createMenuBar()
        self.setWindowTitle("I-BALLAST: Result Visualization Tool")

    def _createMenuBar(self):
        menuBar = QtWidgets.QMenuBar(self)

        fileMenu = QtWidgets.QMenu("&File", self)

        openAction = QtGui.QAction("&Open Result Output Folder", self)
        openAction.triggered.connect(self.openResultFolder)
        fileMenu.addAction(openAction)

        loadAction = QtGui.QAction("&Load GPS Data", self)
        loadAction.triggered.connect(self.loadGPSData)
        fileMenu.addAction(loadAction)

        menuBar.addMenu(fileMenu)
        self.setMenuBar(menuBar)

    def loadGPSData(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fn, filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load GPS data",
            ".",
            "*.csv",
            options=options,
        )
        self.mainWidget.load_gps_from_cvs(fn)

    def openResultFolder(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Result Output Folder", ".", options
        )

        self.mainWidget.loadResults(folder)


class ResultVisualizationTool(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ResourcePool()

        self.mainWindow = ResultVisualizationMainWindow(self.pool)
        self.aboutToQuit.connect(self.cleanup)

    def exec(self):
        self.mainWindow.show()
        super().exec()

    def cleanup(self):
        self.mainWindow.mainWidget.cleanup()
        self.pool.terminate_proxy()
        self.pool = None


if __name__ == "__main__":
    app = ResultVisualizationTool(sys.argv)
    sys.exit(app.exec())
