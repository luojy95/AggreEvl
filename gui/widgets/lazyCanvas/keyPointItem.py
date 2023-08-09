from PySide6 import QtCore

from gui.widgets.taggedQGraphicsPoint import TaggedQGraphicsPointItem


class KeyPointIsRemoving(QtCore.QObject):
    signal = QtCore.Signal(object)


class KeyPointItem(TaggedQGraphicsPointItem):
    def __init__(self, point: QtCore.QPointF, id: int):
        self.id = id
        super().__init__(point, self.getTag())

        self.isRemoving = KeyPointIsRemoving()

    def getTag(self):
        return f"key point: {self.id:03d}"

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self.isRemoving.signal.emit(self)

    def mouseDoubleClickEvent(self, event) -> None:
        self.isRemoving.signal.emit(self)
