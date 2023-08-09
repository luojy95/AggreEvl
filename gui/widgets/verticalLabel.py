from PySide6 import QtGui
from PySide6 import QtWidgets
from PySide6 import QtCore


class VerticalLabel(QtWidgets.QLabel):
    def __init__(self, *args):
        QtWidgets.QLabel.__init__(self, *args)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        # calculate the size of the font
        fm = QtGui.QFontMetrics(painter.font())
        xoffset = int(fm.boundingRect(self.text()).width() / 2)
        yoffset = int(fm.boundingRect(self.text()).height() / 2)
        x = int(self.width() / 2) + yoffset
        y = int(self.height() / 2) - xoffset
        # because we rotated the label, x affects the vertical placement, and y affects the horizontal
        painter.drawText(y, x, self.text())
        painter.end()

    def minimumSizeHint(self):
        size = QtWidgets.QLabel.minimumSizeHint(self)
        return QtCore.QSize(size.height(), size.width())

    def sizeHint(self):
        size = QtWidgets.QLabel.sizeHint(self)
        return QtCore.QSize(size.height(), size.width())
