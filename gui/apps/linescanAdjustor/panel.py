import sys
import numpy as np
from typing import Optional
import PySide6.QtCore

sys.path.append(".")

from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
)

from PySide6.QtWidgets import QSlider, QRadioButton, QCheckBox, QLabel
from PySide6.QtGui import QFont

from gui.widgets.lazyCanvas.lazyCanvas import LazyCanvas, LazySequentialCanvas
from gui.widgets.lazyCanvas.config.lazyGraphicsPixmapItemConfig import (
    LazyGraphicsPixmapItemConfig,
)
from gui.widgets.lazyCanvas.config.common import Crop

# from gui.test.rect import test
from gui.test.lazypixmap import test_projected_gocator, test_linescan
from gui.resourcepool import ResourcePool


def map_range(rate, min_value, max_value):
    assert max_value > min_value
    rate = max(min(rate, 1.0), 0.0)

    return rate * (max_value - min_value) + min_value


def inv_map_range(value, min_value, max_value):
    assert max_value > min_value

    value = max(min(max_value, value), min_value)

    return (value - min_value) / (max_value - min_value)


class LineScanAdjustorPanel(QWidget):
    MIN_BRIGHTNESS = 0.5
    MAX_BRIGHTNESS = 3

    MIN_CONTRAST = 0.5
    MAX_CONTRAST = 2

    MIN_TEMPERATURE = 1000
    MAX_TEMPERATURE = 10000
    TEMPERATURE_TICKS = 500

    MIN_SCALE_Y = 0.5
    MAX_SCALE_Y = 3.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout()

        # title font
        title_font = QFont()
        title_font.setBold(True)

        # spacing between settings
        spacing = 30

        # brightness adjustment
        self.brightness_title_label = QLabel("Brightness", self)
        self.brightness_title_label.setFont(title_font)
        self.brightness_slider = QSlider(
            self, orientation=PySide6.QtCore.Qt.Orientation.Horizontal
        )
        self.brightness_slider.setValue(
            100.0 * inv_map_range(1.0, self.MIN_BRIGHTNESS, self.MAX_BRIGHTNESS)
        )
        self.brightness_label = QLabel(f"{1.0:.2f}", self)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        vbox = QVBoxLayout()
        vbox.addWidget(self.brightness_title_label)
        hbox = QHBoxLayout()
        hbox.addWidget(self.brightness_slider)
        hbox.addWidget(self.brightness_label)
        vbox.addLayout(hbox)
        vbox.addSpacing(spacing)
        layout.addLayout(vbox)

        # contrast adjustment
        self.contrast_title_label = QLabel("Contrast", self)
        self.contrast_title_label.setFont(title_font)
        self.contrast_slider = QSlider(
            self, orientation=PySide6.QtCore.Qt.Orientation.Horizontal
        )
        self.contrast_slider.setValue(
            100.0 * inv_map_range(1.0, self.MIN_CONTRAST, self.MAX_CONTRAST)
        )
        self.contrast_label = QLabel(f"{1.0:.2f}", self)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        vbox = QVBoxLayout()
        vbox.addWidget(self.contrast_title_label)
        hbox = QHBoxLayout()
        hbox.addWidget(self.contrast_slider)
        hbox.addWidget(self.contrast_label)
        vbox.addLayout(hbox)
        vbox.addSpacing(spacing)
        layout.addLayout(vbox)

        # temperature adjustment
        self.temperature_title_label = QLabel("Color Temperature", self)
        self.temperature_title_label.setFont(title_font)

        self.apply_temperature_button = QCheckBox(
            "Apply Color Temperature Setting", self
        )
        self.apply_temperature_button.toggled.connect(self.apply_temperature)
        self.apply_temperature_button.setChecked(False)

        self.temperatures_label = QLabel(f"{self.MAX_TEMPERATURE:5d}K", self)
        self.temperature_slider = QSlider(
            self, orientation=PySide6.QtCore.Qt.Orientation.Horizontal
        )
        self.temperature_slider.setMinimum(self.MIN_TEMPERATURE)
        self.temperature_slider.setMaximum(self.MAX_TEMPERATURE)
        self.temperature_slider.setValue(self.MAX_TEMPERATURE)
        self.temperature_slider.valueChanged.connect(self.adjust_temperature)
        vbox = QVBoxLayout()
        vbox.addWidget(self.temperature_title_label)
        vbox.addWidget(self.apply_temperature_button)
        hbox = QHBoxLayout()
        hbox.addWidget(self.temperature_slider)
        hbox.addWidget(self.temperatures_label)
        vbox.addLayout(hbox)
        vbox.addSpacing(spacing)
        layout.addLayout(vbox)

        # scale Y
        self.scale_title_label = QLabel("Vertical Scale", self)
        self.scale_title_label.setFont(title_font)
        self.scale_slider = QSlider(
            self, orientation=PySide6.QtCore.Qt.Orientation.Horizontal
        )
        self.scale_slider.setValue(
            100.0 * inv_map_range(1.0, self.MIN_SCALE_Y, self.MAX_SCALE_Y)
        )
        self.scale_label = QLabel(f"{1.0:.2f}", self)
        self.scale_slider.valueChanged.connect(self.adjust_scale_y)
        vbox = QVBoxLayout()
        vbox.addWidget(self.scale_title_label)
        hbox = QHBoxLayout()
        hbox.addWidget(self.scale_slider)
        hbox.addWidget(self.scale_label)
        vbox.addLayout(hbox)
        vbox.addSpacing(spacing)
        layout.addLayout(vbox)

        self.setLayout(layout)

        self.canvas: LazyCanvas = None

        self.config: LazyGraphicsPixmapItemConfig = LazyGraphicsPixmapItemConfig()

    def connect_canvas(self, canvas: LazyCanvas):
        self.canvas = canvas

    def update_config(
        self, brightness=None, colortemp=None, contrast=None, horicrop=None
    ):
        brightness = brightness if brightness else self.config.brightness
        colortemp = colortemp if colortemp else self.config.colortemp
        contrast = contrast if contrast else self.config.contrast
        horicrop = horicrop if horicrop else self.config.horicrop

        self.config = LazyGraphicsPixmapItemConfig(
            brightness=brightness,
            contrast=contrast,
            colortemp=colortemp,
            horicrop=horicrop,
        )
        return self.config

    def adjust_brightness(self, value):
        if self.canvas is None:
            return
        brightness_factor = map_range(
            value / 100.0, self.MIN_BRIGHTNESS, self.MAX_BRIGHTNESS
        )

        self.brightness_label.setText(f"{brightness_factor:.2f}")
        self.canvas.view.update_pixmap_config(
            self.update_config(brightness=brightness_factor)
        )

    def adjust_contrast(self, value):
        if self.canvas is None:
            return
        contrast_factor = map_range(value / 100.0, self.MIN_CONTRAST, self.MAX_CONTRAST)
        self.contrast_label.setText(f"{contrast_factor:.2f}")
        self.canvas.view.update_pixmap_config(
            self.update_config(contrast=contrast_factor)
        )

    def adjust_temperature(self, value):
        if self.canvas is None:
            return
        if not self.apply_temperature_button.isChecked():
            self.apply_temperature_button.setChecked(True)
        else:
            correct_value = lambda value: value - value % 500
            self.temperature_slider.setValue(correct_value(value))
            self.temperatures_label.setText(f"{correct_value(value):5d}K")
            self.canvas.view.update_pixmap_config(
                self.update_config(colortemp=correct_value(value))
            )

    def apply_temperature(self, value):
        if self.canvas is None:
            return

        if not value:
            self.canvas.view.update_pixmap_config(self.update_config(colortemp=-1))
            self.temperature_slider.setDisabled(True)
        else:
            self.temperature_slider.setDisabled(False)
            self.adjust_temperature(self.temperature_slider.value())

    def adjust_scale_y(self, value):
        # TODO: scale y logic
        if self.canvas is None:
            return

        scale_factor = map_range(value / 100.0, self.MIN_SCALE_Y, self.MAX_SCALE_Y)
        self.scale_label.setText(f"{scale_factor:.2f}")

        self.canvas.view.update_scale_y(scale_factor)

    def adjust_hori_crop(self, crop: Crop):
        if self.canvas is None:
            return

        self.canvas.view.update_pixmap_config(self.update_config(horicrop=crop))
