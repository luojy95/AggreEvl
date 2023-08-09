from PySide6 import QtCore

from gui.widgets.taggedQGraphicsLine import TaggedQGraphicsLineItem


class KeyLineIsRemoving(QtCore.QObject):
    signal = QtCore.Signal(object)


class KeyLineItem(TaggedQGraphicsLineItem):
    def __init__(self, line: QtCore.QLine, id: int):
        self.id = id
        super().__init__(line, self.getTag())

        self.isRemoving = KeyLineIsRemoving()

    def getTag(self):
        return f"key line: {self.id:03d}"

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self.isRemoving.signal.emit(self)

    def mouseDoubleClickEvent(self, event) -> None:
        self.isRemoving.signal.emit(self)
