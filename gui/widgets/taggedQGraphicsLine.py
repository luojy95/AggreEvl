from typing import Optional
from PySide6 import QtWidgets, QtGui, QtCore


class TaggedQGraphicsLineItem(QtWidgets.QGraphicsLineItem):
    def __init__(self, line: QtCore.QLine, tag: str):
        self.tag = tag
        super().__init__(line)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget,
    ) -> None:
        super().paint(painter, option, widget)
        fm = QtGui.QFontMetrics(painter.font())
        yoffset = int(fm.boundingRect(self.tag).height() / 2)
        xoffset = self.scene().views()[0].horizontalScrollBar().value()
        scale = self.scene().views()[0].current_zoom
        painter.drawText(
            self.line().x1() + xoffset / scale,
            self.line().y1() - yoffset,
            self.tag,
        )
