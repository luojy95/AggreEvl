import copy
from typing import Optional
import PySide6.QtWidgets
from PySide6.QtWidgets import QStyleOptionGraphicsItem, QWidget
from gui.resourcepool import ResourcePool
from gui.widgets.lazyCanvas.lazyGraphicsItem import (
    LazyGraphicsPixmapItem,
    LazyInterface,
)
from PySide6 import QtGui, QtWidgets, QtCore


class ItemChanged(QtCore.QObject):
    signal = QtCore.Signal(object)


class LinkedLazyGraphicsItem(LazyGraphicsPixmapItem):
    def __init__(self, linked_item: LazyGraphicsPixmapItem, parent=None) -> None:
        self.linked_item = linked_item
        assert self.linked_item != None
        super().__init__(
            self.linked_item.size.width(),
            self.linked_item.size.height(),
            self.linked_item.raw_size.width(),
            self.linked_item.raw_size.height(),
            self.linked_item.parent,
        )
        self.linked_item.linked_item_connect.presentPixmap.connect(
            self.updateWithQpixmap
        )
        self.alpha = 1.0
        self.item_changed = ItemChanged()
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.config_pos_offset = QtCore.QPointF(0.0, 0.0)
        self.config_scale = QtCore.QPointF(1.0, 1.0)
        self.override_pos_offset = QtCore.QPointF(0.0, 0.0)
        self.override_scale = QtCore.QPointF(1.0, 1.0)

    def scaled_size(self):
        width = self.linked_item.size.width()
        if self.override_scale.x() * self.config_scale.x() != 1.0:
            width *= self.override_scale.x() * self.config_scale.x()
            width = int(width) + ((width - int(width)) > 0.01)

        height = self.linked_item.size.height()
        if self.override_scale.y() * self.config_scale.y() != 1.0:
            height *= self.override_scale.y() * self.config_scale.y()
            height = int(height) + ((height - int(height)) > 0.01)

        return QtCore.QSize(width, height)

    def final_pos(self):
        return (
            self.linked_item.pos() + self.config_pos_offset + self.override_pos_offset
        )

    def paint(
        self, painter: QtGui.QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ) -> None:
        painter.setOpacity(self.alpha)
        super(LazyGraphicsPixmapItem, self).paint(painter, option, widget)
        painter.save()
        painter.setOpacity(1.0)
        pen = QtGui.QPen()
        pen.setColor("black")
        painter.setBrush(QtGui.QColor(0, 0, 0, 255))
        ratio = self.size.width() / self.raw_size.width()
        if self.pixmap_config.horicrop.left > 0:
            left_rect = QtCore.QRectF(
                0,
                0,
                self.pixmap_config.horicrop.left * ratio,
                self.size.height(),
            )
            painter.drawRect(left_rect)
        if self.pixmap_config.horicrop.right > 0:
            right_rect = QtCore.QRectF(
                self.size.width() - self.pixmap_config.horicrop.right * ratio,
                0,
                self.pixmap_config.horicrop.right * ratio,
                self.size.height(),
            )
            painter.drawRect(right_rect)
        painter.restore()

    def load(self, resourcepool=None):
        self.linked_item.load(resourcepool)

    def release(self, resourcepool=None):
        self.qimage = None
        self.qimage_preview = None

    # def release(
    #     self, resourcepool: ResourcePool = None, erase_pixmap=True, erase_from_pool=True
    # ):
    # return super().release(resourcepool, erase_pixmap, erase_from_pool)
    # return

    def updateOverride(self):
        if self is None:
            return
        if self.qpixmap is not None:
            if self.final_pos() != self.pos() or self.scaled_size() != self.size:
                self.size = self.scaled_size()
                if self.qpixmap.isNull():
                    self.qpixmap = QtGui.QPixmap(self.scaled_size())
                    self.qpixmap.fill(QtGui.QColor("black"))
                else:
                    self.qpixmap = self.qpixmap.scaled(
                        self.scaled_size(),
                        QtCore.Qt.IgnoreAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )

                # if self.qimage_preview is not None:
                #     qimage = self.qimage_preview.scaled(
                #         self.scaled_size(), QtCore.Qt.IgnoreAspectRatio
                #     )
                #     self.qpixmap.convertFromImage(qimage)

                self.setPos(self.final_pos())

            self.setPixmap(self.qpixmap)

    def updateWithQpixmap(self, qpixmap: QtGui.QPixmap):
        if self is None:
            return

        if qpixmap is not None:
            if self.final_pos() != self.pos() or self.scaled_size() != self.size:
                self.size = self.scaled_size()
                self.qpixmap = qpixmap.scaled(
                    self.scaled_size(),
                    QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.setPixmap(self.qpixmap)
                self.setPos(self.final_pos())
                self.item_changed.signal.emit(self)
            else:
                self.qpixmap = qpixmap.scaled(
                    self.scaled_size(),
                    QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.setPixmap(self.qpixmap)
