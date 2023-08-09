import sys

sys.path.append(".")

import os
import yaml

import PySide6.QtCore as QtCore
from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QGridLayout,
)
from PySide6.QtGui import QKeyEvent, QWheelEvent

from gui.widgets.lazyCanvas.lazyCanvas import LazySequentialCanvas
from gui.apps.linescanAdjustor.panel import LineScanAdjustorPanel
from gui.widgets.rangeSlider import RangeSlider
from gui.widgets.verticalLabel import VerticalLabel
from gui.widgets.lazyCanvas.config.common import Range, Crop
from gui.test.lazypixmap import test_linescan
from gui.resourcepool import ResourcePool

from tools.logger import default_logger as logger


class LineScanAdjustorWidget(QWidget):
    MINIMUM_WIDTH_CANVAS = 800
    MINIMUM_HEIGHT_CANVAS = 600
    MINIMUM_WIDTH_PANEL = 400

    def __init__(self, resourcepool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.pool = resourcepool

        self.displayRangeSlider = RangeSlider(
            orientation=QtCore.Qt.Orientation.Vertical
        )
        self.displayRangeSlider.rangeChanged.connect(self.onDisplayRangeChanged)
        self.displayRangeLabel = VerticalLabel("Image Count", self)
        self.displayRangeLabelTop = QLabel(f"{0:d}", self)
        self.displayRangeLabelTop.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.displayRangeLabelBottom = QLabel(f"{0:d}", self)
        self.displayRangeLabelBottom.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        displayRangeSliderHbox = QHBoxLayout()
        displayRangeSliderHbox.addWidget(self.displayRangeLabel)
        displayRangeSliderHbox.addWidget(self.displayRangeSlider)
        displayRangeVbox = QVBoxLayout()
        displayRangeVbox.addWidget(self.displayRangeLabelTop)
        displayRangeVbox.addLayout(displayRangeSliderHbox)
        displayRangeVbox.addWidget(self.displayRangeLabelBottom)

        self.horizontalCropSlider = RangeSlider()
        self.horizontalCropSlider.rangeChanged.connect(self.onHorizontalCropChanged)
        self.horizontalCropLabel = QLabel("Horizontal Crop (Pixel)", self)
        self.horizontalCropRangeLabelLeft = QLabel(f"{0:d}", self)
        self.horizontalCropRangeLabelRight = QLabel(f"{0:d}", self)
        horizontalSliderHbox = QHBoxLayout()
        horizontalSliderHbox.addWidget(self.horizontalCropRangeLabelLeft)
        horizontalSliderHbox.addWidget(self.horizontalCropSlider)
        horizontalSliderHbox.addWidget(self.horizontalCropRangeLabelRight)
        horizontalLabelHbox = QHBoxLayout()
        horizontalLabelHbox.addStretch()
        horizontalLabelHbox.addWidget(self.horizontalCropLabel)
        horizontalLabelHbox.addStretch()
        horizontalCropVbox = QVBoxLayout()
        horizontalCropVbox.addLayout(horizontalSliderHbox)
        horizontalCropVbox.addLayout(horizontalLabelHbox)

        self.linescan_canvas: LazySequentialCanvas = LazySequentialCanvas(self.pool)
        self.linescan_canvas.setMinimumHeight(self.MINIMUM_HEIGHT_CANVAS)
        self.linescan_canvas.setMinimumWidth(self.MINIMUM_WIDTH_CANVAS)

        left_layout = QGridLayout()
        left_layout.addLayout(displayRangeVbox, 0, 0)
        left_layout.addWidget(self.linescan_canvas, 0, 1)
        left_layout.addLayout(horizontalCropVbox, 1, 1)

        layout.addLayout(left_layout, 2)

        right_layout = QVBoxLayout()
        self.adjustor_panel = LineScanAdjustorPanel()
        self.adjustor_panel.connect_canvas(self.linescan_canvas)
        self.adjustor_panel.setMinimumWidth(self.MINIMUM_WIDTH_PANEL)

        self.save_button = QPushButton("&Save Settings", parent=self)
        self.save_button.clicked.connect(self.onClickSaveButton)

        right_layout.addWidget(self.adjustor_panel)
        right_layout.addWidget(self.save_button)
        right_layout.addStretch()

        layout.addLayout(right_layout, 1)

        self.setLayout(layout)

        self.workspace = None

        self.current_settings_file = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        return self.linescan_canvas.view.keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        return self.linescan_canvas.view.keyReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        return self.linescan_canvas.view.wheelEvent(event)

    def cleanup(self) -> None:
        logger.info("[LineScanAdjustor]Clean up tmp files")
        self.linescan_canvas.clearSceneItems()

    def load_data(self, path):
        self.cleanup()
        try:
            test_linescan(self.linescan_canvas, target_path=path)
            self.reset_sliders()
            self.linescan_canvas.view.initSceneRect()
            self.workspace = path
        except:
            self.cleanup()

    def reset_sliders(self):
        items = self.linescan_canvas.view.scene().item_seq
        num_items = len(items)
        self.displayRangeSlider.setRangeLimit(0, num_items)
        self.displayRangeSlider.setRange(0, num_items)
        self.displayRangeLabel.setText(f"Selected Image Count: {num_items}")
        self.displayRangeLabelTop.setText(f"{0}")
        self.displayRangeLabelBottom.setText(f"{num_items}")

        max_width = 0
        if num_items > 0:
            first_item = items[0]
            max_width = first_item.raw_size.width()
        self.horizontalCropSlider.setRangeLimit(0, max_width)
        self.horizontalCropSlider.setRange(0, max_width)
        self.horizontalCropRangeLabelLeft.setText(f"{0}")
        self.horizontalCropRangeLabelRight.setText(f"{max_width}")

    def onDisplayRangeChanged(self, current_range):
        displayRange = Range(current_range[0], current_range[1])
        view_range = self.linescan_canvas.scene.maskDisplayWithRange(displayRange)
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
        self.linescan_canvas.view.view_config.displayRange = displayRange
        self.linescan_canvas.view.update_crop_rect()
        self.displayRangeLabelTop.setText(f"{displayRange.start}")
        self.displayRangeLabelBottom.setText(f"{displayRange.end}")
        self.displayRangeLabel.setText(
            f"Selected Image Count: {displayRange.end - displayRange.start}"
        )

    def onHorizontalCropChanged(self, current_range):
        horizontalCrop = Crop(
            left=current_range[0],
            right=self.horizontalCropSlider.opt.maximum - current_range[1],
        )
        self.horizontalCropRangeLabelLeft.setText(f"{current_range[0]}")
        self.horizontalCropRangeLabelRight.setText(f"{current_range[1]}")
        self.adjustor_panel.adjust_hori_crop(horizontalCrop)

    def export_settings(self, path):
        data = dict(
            workspace=self.workspace,
            displayRange=dict(
                start=self.linescan_canvas.view.view_config.displayRange.start,
                end=self.linescan_canvas.view.view_config.displayRange.end,
            ),
            horizontalCrop=dict(
                left=self.linescan_canvas.view.pixmap_config.horicrop.left,
                right=self.linescan_canvas.view.pixmap_config.horicrop.right,
            ),
            horizontalFlip=self.linescan_canvas.view.pixmap_config.horiflip,
            initialScaleY=self.linescan_canvas.view.view_config.initialScale.y,
            brightness=self.linescan_canvas.view.pixmap_config.brightness,
            colorTemperature=self.linescan_canvas.view.pixmap_config.colortemp,
            contrast=self.linescan_canvas.view.pixmap_config.contrast,
        )

        with open(path, "w+") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        logger.info(f"[LineScanAdjustor]Exported settings to {path}")

    def load_settings(self, path):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

                self.displayRangeSlider.setRange(
                    data["displayRange"]["start"], data["displayRange"]["end"]
                )
                self.horizontalCropSlider.setRange(
                    data["horizontalCrop"]["left"],
                    self.horizontalCropSlider.opt.maximum
                    - data["horizontalCrop"]["right"],
                )

                def parse(value, minv, maxv):
                    return (value - minv) / (maxv - minv) * 100

                self.adjustor_panel.scale_slider.setValue(
                    parse(
                        data["initialScaleY"],
                        self.adjustor_panel.MIN_SCALE_Y,
                        self.adjustor_panel.MAX_SCALE_Y,
                    )
                )
                if data["colorTemperature"] == -1:
                    self.adjustor_panel.apply_temperature_button.setChecked(False)
                else:
                    self.adjustor_panel.apply_temperature_button.setChecked(True)
                    self.adjustor_panel.temperature_slider.setValue(
                        data["colorTemperature"]
                    )
                self.adjustor_panel.brightness_slider.setValue(
                    parse(
                        data["brightness"],
                        self.adjustor_panel.MIN_BRIGHTNESS,
                        self.adjustor_panel.MAX_BRIGHTNESS,
                    )
                )
                self.adjustor_panel.contrast_slider.setValue(
                    parse(
                        data["contrast"],
                        self.adjustor_panel.MIN_CONTRAST,
                        self.adjustor_panel.MAX_CONTRAST,
                    )
                )
                self.current_settings_file = path
        except yaml.YAMLError as exc:
            print(exc)

    def onClickSaveButton(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fn, ext = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Line Scan Settings",
            "line_scan_settings"
            if self.current_settings_file is None
            else os.path.splitext(self.current_settings_file)[0],
            ".yaml",
            options=options,
        )

        self.export_settings(fn + ext)


class LineScanAdjustorMainWindow(QMainWindow):
    def __init__(self, resourcepool, parent=None) -> None:
        super().__init__(parent)
        self.pool = resourcepool

        self.mainWidget = LineScanAdjustorWidget(self.pool)
        self.setCentralWidget(self.mainWidget)

        self._createMenuBar()

        self.current_work_folder = None

        self.setWindowTitle("I-BALLAST: Line Scan Camera Image Adjustor")

    def _createMenuBar(self):
        menuBar = QtWidgets.QMenuBar(self)

        fileMenu = QtWidgets.QMenu("&File", self)

        openAction = QtGui.QAction("&Open Line Scan Image Folder", self)
        openAction.triggered.connect(self.openFolder)
        fileMenu.addAction(openAction)

        loadAction = QtGui.QAction("&Load Line Scan Settings", self)
        loadAction.triggered.connect(self.loadSettings)
        fileMenu.addAction(loadAction)

        menuBar.addMenu(fileMenu)
        self.setMenuBar(menuBar)

    def openFolder(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_dialog = QtWidgets.QFileDialog(self, options=options)
        file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if file_dialog.exec() == QtWidgets.QFileDialog.Accepted:
            folder_path = file_dialog.selectedFiles()[0]
            logger.info(f"[LineScanAdjustor]Open directory {folder_path}")
            if os.path.isdir(folder_path) and os.path.exists(folder_path):
                self.current_work_folder = folder_path
                self.mainWidget.load_data(self.current_work_folder)

    def loadSettings(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fn, filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Line Scan Settings",
            ".",
            "*.yaml",
            options=options,
        )
        self.mainWidget.load_settings(fn)


class LineScanAdjustor(QApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ResourcePool()
        # self.setApplicationDisplayName("I-BALLAST: Line Scan Camera Image Adjustor")
        self.mainWindow: LineScanAdjustorMainWindow = LineScanAdjustorMainWindow(
            self.pool
        )
        self.aboutToQuit.connect(self.cleanup)

    def exec(self):
        self.mainWindow.show()
        super().exec()

    def cleanup(self):
        self.mainWindow.mainWidget.cleanup()
        self.pool.terminate_proxy()
        self.pool = None


if __name__ == "__main__":
    app = LineScanAdjustor(sys.argv)
    sys.exit(app.exec())
