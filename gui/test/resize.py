import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class GraphItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, xdata, ydata, width, height, parent=None):
        super(GraphItem, self).__init__(parent)

        self._xdata = xdata
        self._ydata = ydata
        self._size = QtCore.QSize(width, height)
        self.draw()

    def draw(self):
        x_final = self._xdata[-1]
        pixmap = QtGui.QPixmap(self._size)
        pixmap_height = pixmap.height()
        pixmap.fill(QtGui.QColor("lightblue"))
        painter = QtGui.QPainter(pixmap)

        pen = QtGui.QPen(QtGui.QColor("green"))
        pen.setWidth(2)
        painter.setPen(pen)
        for i, (x, y) in enumerate(
            zip(self._xdata, self._ydata / np.max(np.abs(self._ydata)))
        ):
            x_pos = int(x * self._size.width() / x_final)
            y_pos = abs(int(y * pixmap_height))
            painter.drawLine(x_pos, 0, x_pos, y_pos)

        painter.end()
        self.setPixmap(pixmap)


class HorizontalRectItem(QtWidgets.QGraphicsRectItem):
    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.scene():
            newPos = self.pos()
            newPos.setX(value.x())
            return newPos
        return super(HorizontalRectItem, self).itemChange(change, value)


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self.scale(1, -1)

    def resizeEvent(self, event):
        h = self.mapToScene(self.viewport().rect()).boundingRect().height()
        r = self.sceneRect()
        r.setHeight(h)
        self.setSceneRect(r)

        height = self.viewport().height()
        for item in self.items():
            item_height = item.boundingRect().height()
            tr = QtGui.QTransform()
            tr.scale(1, height / item_height)
            item.setTransform(tr)

        super(GraphicsView, self).resizeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        tab_widget = QtWidgets.QTabWidget(tabPosition=QtWidgets.QTabWidget.West)
        self.setCentralWidget(tab_widget)

        self.graphics_view_top = GraphicsView()
        self.graphics_view_bottom = QtWidgets.QGraphicsView()

        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)
        lay.addWidget(self.graphics_view_top)
        lay.addWidget(self.graphics_view_bottom)

        tab_widget.addTab(container, "main")

        self.resize(640, 480)

        side, offset, height = 50, 200, 400

        np.random.seed(777)
        x_time = np.linspace(0, 12.56, 3000)
        rand_data = np.random.uniform(0.0, 1.0, 3000)
        data = 0.45 * (np.sin(2 * x_time) + rand_data) - 0.25 * (np.sin(3 * x_time))

        graph_item = GraphItem(x_time, data, 3000, height)
        self.graphics_view_top.scene().addItem(graph_item)

        for i in range(2):
            r = QtCore.QRectF(
                QtCore.QPointF((i + 1) * offset + i * 2 * side, 2),
                QtCore.QSizeF(side, height),
            )
            it = HorizontalRectItem(r)
            it.setPen(QtGui.QPen(QtGui.QColor("red"), 2))
            it.setBrush(QtGui.QColor(255, 0, 0, 127))
            self.graphics_view_top.scene().addItem(it)
            it.setFlags(
                it.flags()
                | QtWidgets.QGraphicsItem.ItemIsMovable
                | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
            )


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()

    ret = app.exec()

    sys.exit(ret)


if __name__ == "__main__":
    main()
