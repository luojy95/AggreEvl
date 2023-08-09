from PySide6 import QtCore, QtWidgets, QtGui
import PySide6.QtCore
import PySide6.QtGui
from PIL import ImageColor
import numpy as np


class ColorBarLabel(QtWidgets.QLabel):
    PADDING = 10

    COLOR_RECT_SIZE = 20

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors = []
        for color in colors:
            qcolor = self.parseRawColor(color)
            if qcolor is not None:
                self.colors.append(qcolor)

        self.values = [0.0 for color in self.colors]
        self.total_width = 0

    def parseRawColor(self, color) -> QtGui.QColor:
        if type(color) == str:
            rgb = ImageColor.getrgb(color)
            if rgb is None:
                return None
            else:
                return QtGui.QColor(rgb[0], rgb[1], rgb[2], 255)
        elif isinstance(color, QtGui.QColor):
            return color

        elif isinstance(color, np.ndarray):
            return QtGui.QColor(color[0], color[1], color[2], 255)

        return None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        start_x = 0
        for i in range(len(self.colors)):
            start_x += self.PADDING
            item_width = self.drawColor(painter, i, start_x)
            start_x += item_width
        painter.end()
        self.total_width = start_x
        self.setMinimumHeight(self.sizeHint().height())

    def drawColor(self, painter: QtGui.QPainter, colorid: int, start_x: int):
        color_item_width = 0
        color: QtGui.QColor = self.colors[colorid]
        value: float = self.values[colorid]
        painter.save()
        painter.setBrush(color)
        rect = QtCore.QRectF(
            start_x, self.PADDING, self.COLOR_RECT_SIZE, self.COLOR_RECT_SIZE
        )
        color_item_width += self.COLOR_RECT_SIZE
        painter.drawRect(rect)
        if value is not None:
            painter.font().setPointSize(self.COLOR_RECT_SIZE)
            fm = QtGui.QFontMetrics(painter.font())
            if type(value) == float:
                tag = f"{round(value)}"
            elif type(value) == str:
                tag = value

            font_height = fm.boundingRect(tag).height()
            font_width = max(
                fm.boundingRect(tag).width(), self.COLOR_RECT_SIZE + self.PADDING
            )
            painter.drawText(
                start_x + self.COLOR_RECT_SIZE + self.PADDING,
                self.PADDING + font_height,
                tag,
            )
            color_item_width += self.PADDING
            color_item_width += font_width
        painter.restore()
        return color_item_width

    def sizeHint(self):
        return QtCore.QSize(self.total_width, self.COLOR_RECT_SIZE + self.PADDING * 2)
