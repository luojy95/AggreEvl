from typing import Optional
from PySide6 import QtWidgets, QtGui, QtCore
import PySide6.QtGui
import PySide6.QtWidgets


class TaggedQGraphicsPointItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, center: QtCore.QPointF, tag: str, size: int = 10):
        self.tag = tag
        self.center = center
        half_size = size // 2
        rectf = QtCore.QRectF(
            center.x() - half_size,
            center.y() - half_size,
            size,
            size,
        )
        super().__init__(rectf)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget,
    ) -> None:
        super().paint(painter, option, widget)

        fm = QtGui.QFontMetrics(painter.font())
        font_height = fm.boundingRect(self.tag).height()
        font_width = fm.boundingRect(self.tag).width()
        padding = 2

        painter.save()
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0.0, 0.0, 0.0, 200))
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0.0, 0.0, 0.0, 200))

        painter.drawRoundedRect(
            QtCore.QRectF(
                self.center.x(),
                self.center.y() - padding - font_height,
                2 * padding + font_width,
                2 * padding + font_height,
            ),
            2,
            2,
        )

        painter.restore()

        # scale = self.scene().views()[0].current_zoom
        # xoffset = self.scene().views()[0].horizontalScrollBar().value() / scale

        painter.drawText(
            self.center.x() + padding,
            self.center.y() - padding,
            self.tag,
        )
