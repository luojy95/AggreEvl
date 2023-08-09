from typing import Optional
from PySide6 import QtWidgets, QtGui, QtCore
import PySide6.QtGui
import PySide6.QtWidgets


class TaggedQGraphicsRectItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, rect: QtCore.QRectF, tag: str):
        self.tag = tag
        self._rect = rect
        super().__init__(rect)

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
                self.rect().right() + padding,
                self.rect().top() - padding - font_height,
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
            self.rect().right() + 2 * padding,
            self.rect().top() - padding,
            self.tag,
        )
