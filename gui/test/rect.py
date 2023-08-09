import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsRectItem
from PySide6.QtGui import QBrush, QPen

from gui.widgets.lazyCanvas.lazyCanvas import LazyCanvas


def create_rect_item(x, y, w, h):
    rect = QGraphicsRectItem(x, y, w, h)
    rect.setBrush(QBrush(Qt.GlobalColor.cyan))
    rect.setPen(QPen(Qt.GlobalColor.red))
    return rect


def test(canvas: LazyCanvas):
    num_rects = 100000
    for i in range(num_rects):
        canvas.addItem(create_rect_item(0, 50 * i, 4096, 50))
