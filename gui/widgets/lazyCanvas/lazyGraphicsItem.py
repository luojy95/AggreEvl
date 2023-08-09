import os
from typing import Any, List
from PySide6 import Qt, QtGui
from PySide6.QtCore import QSize, QRect, Signal, QObject, QPointF
from PySide6 import QtCore
from PySide6.QtWidgets import (
    QWidget,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QStyleOptionGraphicsItem,
    QGraphicsSceneMouseEvent,
)

import uuid
from PIL import Image, ImageEnhance, ImageQt

from gui.qimgutils import qImgUtils
from gui.resourcepool import ResourcePool
from gui.widgets.lazyCanvas.config.lazyGraphicsPixmapItemConfig import (
    LazyGraphicsPixmapItemConfig,
)
from gui.widgets.lazyCanvas.common import *
from gui.widgets.lazyCanvas.config.common import Crop

from tools.imageprocess import enhance_brightness, enhance_contrast, convert_temperature
import tempfile


class LazyInterface:
    def __init__(self, parent=None) -> None:
        pass

    def load(self, resourcepool=None):
        raise NotImplementedError

    def release(self, resourcepool=None):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SceneItemConnect(QObject):
    position_change_signal = Signal()


class LinkedItemConnect(QObject):
    presentPixmap: Signal = Signal(object)


class LazyGraphicsPixmapItem(QGraphicsPixmapItem, LazyInterface):
    def __init__(self, width, height, original_width, original_height, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.uuid = uuid.uuid4()

        self.sourcePath = None

        self.path = None

        # do not use tmpFile directly!
        self.tmpFilePath = None
        self.size = QSize(width, height)
        self.raw_size = QSize(original_width, original_height)
        self.qimage: QtGui.QImage = None
        self.qimage_preview: QtGui.QImage = None
        self.qpixmap = QtGui.QPixmap(self.size)
        self.qpixmap.fill(QtGui.QColor("black"))
        self.setPixmap(self.qpixmap)
        self.last_time_mouse_position = None
        self.scene_item_connect: SceneItemConnect = SceneItemConnect()
        self.linked_item_connect: LinkedItemConnect = LinkedItemConnect()
        self.setZValue(0)

        self.pixmap_config: LazyGraphicsPixmapItemConfig = (
            LazyGraphicsPixmapItemConfig()
        )
        self.setTransformationMode(QtCore.Qt.SmoothTransformation)

        self.loading = False

    def setPath(self, path):
        self.path = path

    def loadImg(self) -> QtGui.QImage:
        return qImgUtils.load_qimg(self.path)

    def loadTmpImg(self) -> QtGui.QImage:
        return qImgUtils.load_qimg(self.tmpFilePath)

    def update_config(self, config: LazyGraphicsPixmapItemConfig = None):
        self.pixmap_config = config

    def preview_config(self):
        if self.qimage is None:
            return

        if self.pixmap_config != None:
            image = Image.fromqimage(self.qimage)
            enhanced: Image.Image
            enhanced = enhance_brightness(image, self.pixmap_config.brightness)
            enhanced = enhance_contrast(enhanced, self.pixmap_config.contrast)

            if self.pixmap_config.colortemp != -1:
                enhanced = convert_temperature(enhanced, self.pixmap_config.colortemp)

            if self.pixmap_config.horiflip:
                enhanced = enhanced.transpose(Image.FLIP_LEFT_RIGHT)

            if self.pixmap_config.vertflip:
                enhanced = enhanced.transpose(Image.FLIP_TOP_BOTTOM)

            self.qimage_preview = ImageQt.ImageQt(enhanced)

        else:
            self.qimage_preview = self.qimage.copy()
        self.qpixmap.convertFromImage(self.qimage_preview)
        self.linked_item_connect.presentPixmap.emit(self.qpixmap)
        self.setPixmap(self.qpixmap)

    def appy_config(self, resourcepool: ResourcePool):
        self.qimage = self.qimage_preview.copy()
        self.optimize_io(resourcepool)

    def optimize_io(self, resourcepool=None) -> None:
        if self.qimage is not None and self.tmpFilePath is None:
            fd, path = tempfile.mkstemp(prefix="bsv_datakit_lazy_pixmap_item_")
            self.tmpFilePath = path
            self.qimage.save(path, "png")
            os.close(fd)

            if resourcepool:
                resourcepool.release(
                    str(self.uuid) + "@" + self.path, erase_from_pool=True
                )
            # self.path = path

    def load(self, resourcepool=None):
        if self.loading:
            return

        if self.path is None:
            print("resource path cannot be none!")
            raise ValueError()

        self.loading = True

        callback = lambda: self.update(resourcepool)
        if self.tmpFilePath is not None:
            resourcepool.require(
                str(self.uuid) + "@" + self.path, self.loadTmpImg, callback
            )
        else:
            resourcepool.require(
                str(self.uuid) + "@" + self.path, self.loadImg, callback
            )

    def release(
        self,
        resourcepool: ResourcePool = None,
        erase_pixmap=False,
        erase_from_pool=True,
    ):
        self.loading = False
        if erase_pixmap:
            self.qpixmap.fill(QtGui.QColor("black"))
            self.setPixmap(self.qpixmap)
        if resourcepool:
            resourcepool.release(
                str(self.uuid) + "@" + self.path, erase_from_pool=erase_from_pool
            )

    def update(self, resourcepool=None):
        self.loading = False
        if self is None:  # avoid calling update when item was deleted
            return

        can_update = False
        if resourcepool:
            qimage = resourcepool.get(str(self.uuid) + "@" + self.path)
            if qimage:
                can_update = True
                self.qimage = qimage.scaled(self.size, QtCore.Qt.IgnoreAspectRatio)

                self.preview_config()
                # self.qpixmap.convertFromImage(self.qimage_preview)
                # self.setPixmap(self.qpixmap)
                self.optimize_io(resourcepool)

        if not can_update:
            self.qpixmap.fill(QtGui.QColor("black"))
            self.setPixmap(self.qpixmap)

    def clean_tempfile(self):
        if self.tmpFilePath is not None and os.path.exists(self.tmpFilePath):
            os.remove(self.tmpFilePath)

    def vsplit_meta_data(self, newItem, split_point):
        pass

    def vsplit(
        self, split_point: QPointF, resourcepool: ResourcePool
    ) -> "LazyGraphicsPixmapItem":
        if self.qimage is None:
            return None

        if self.tmpFilePath is not None and os.path.exists(self.tmpFilePath):
            os.remove(self.tmpFilePath)

        ratio = split_point.y() * 1.0 / self.size.height()

        qimage_1, qimage_2 = qImgUtils.vsplit(self.qimage, ratio)

        # assign qimage_2 to new Item
        newItem = LazyGraphicsPixmapItem(
            self.size.width(), self.size.height() - split_point.y(), self.parent
        )
        newItem.qimage = qimage_2
        newItem.optimize_io()

        # assign qimage_1 to self
        self.qimage = qimage_1
        self.size = QSize(self.size.width(), split_point.y())
        self.qpixmap = QtGui.QPixmap(self.size)
        self.tmpFilePath = None
        self.optimize_io()
        self.load(resourcepool)

        self.vsplit_meta_data(newItem, split_point)

        # Add new item to the scene
        newItem.setPos(self.x(), self.y() + self.size.height())
        self.scene().addItem(newItem)
        newItem.load(resourcepool)

        return newItem

    def merge_meta_data(self, item_list):
        pass

    @classmethod
    def merge(
        cls, item_list: List["LazyGraphicsPixmapItem"], resourcepool: ResourcePool
    ) -> "LazyGraphicsPixmapItem":
        """generate a new item that combines all the items in the list

        Args:
            item_list (list[&quot;LazyGraphicsPixmapItem&quot;]): _description_

        Returns:
            LazyGraphicsPixmapItem: _description_
        """
        qimage = qImgUtils.vstack([item.qimage for item in item_list])
        # assign qimage_2 to new Item
        newItem = LazyGraphicsPixmapItem(
            qimage.size.width(), qimage.size.height(), item_list[0].parent
        )
        newItem.qimage = qimage
        newItem.optimize_io(resourcepool)
        newItem.load(resourcepool)

        return newItem

    def merge(
        self, item_list: List["LazyGraphicsPixmapItem"], resourcepool: ResourcePool
    ) -> None:
        """merge the current item and items in the item list in place

        Args:
            item_list (list[&quot;LazyGraphicsPixmapItem&quot;]): _description_
        """
        # print(self.size)
        # combine qimages
        qimage = qImgUtils.vstack([self.qimage] + [item.qimage for item in item_list])
        #  inplace, reset self
        # assign qimage_1 to self
        self.qimage = qimage
        self.size = qimage.size()
        self.qpixmap = QtGui.QPixmap(self.size)
        self.tmpFilePath = None
        self.optimize_io(resourcepool)
        self.load(resourcepool)

        for item in item_list[::-1]:
            self.scene().removeItem(item)

    def resize(self, size: QSize, resourcepool: ResourcePool) -> None:
        qimage = qImgUtils.resize(self.qimage, size)
        #  inplace, reset self
        # assign qimage_1 to self
        self.qimage = qimage
        self.size = qimage.size()
        self.qpixmap = QtGui.QPixmap(self.size)
        self.load(resourcepool)

    def paint(
        self, painter: QtGui.QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ) -> None:
        super().paint(painter, option, widget)

        if self.isSelected():
            painter.save()
            path = self.shape()

            pen = QtGui.QPen(
                QtCore.Qt.yellow,
                3,
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            painter.setPen(pen)
            painter.drawPath(path)
            painter.restore()
            self.setZValue(UPMOST_LAYER_Z)
        else:
            self.setZValue(COMMON_ITEM_LAYER_Z)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.isSelected():
            if self.last_time_mouse_position is None:
                self.last_time_mouse_position = event.scenePos()
            else:
                new_mouse_pos = event.scenePos()

                incr = new_mouse_pos - self.last_time_mouse_position

                # self.setPos(self.x(), self.y() + incr.y())

                # self.itemChange(
                #     QGraphicsItem.GraphicsItemChange.ItemPositionChange, incr.y()
                # )

                self.last_time_mouse_position = new_mouse_pos

        # print(new_pos, self.scenePos())

        # Keep the old Y position, so only the X-pos changes.
        # old_pos = self.scenePos()
        # new_pos.setY(old_pos.y())

    # return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.last_time_mouse_position = None

        return super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        """override itemChange event, e.g. selected/unselected

        Args:
            change (QGraphicsItem.GraphicsItemChange): _description_
            value (Any): _description_

        Returns:
            Any: _description_
        """
        # if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
        # self.scene_item_connect.position_change_signal.emit()
        #     if isinstance(self.scene(), SequentialGraphicsScene):
        #         self.scene().setup_items()
        # if change is QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
        # if value:
        #     pass
        # if change != QGraphicsItem.GraphicsItemChange.ItemZValueChange:
        #     print("Changed ", change, value)

        # else:
        #     print("Unselected")
        return super().itemChange(change, value)
