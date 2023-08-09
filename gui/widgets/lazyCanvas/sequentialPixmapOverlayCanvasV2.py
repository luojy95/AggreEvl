from copy import deepcopy
from typing import Optional, Dict
from collections import deque
from bisect import bisect_left
from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np
from sklearn.linear_model import LinearRegression


from gui.widgets.lazyCanvas.lazyCanvas import (
    LazySequentialCanvas,
    LazyCanvas,
    SequentialGraphicsScene,
)
from gui.widgets.lazyCanvas.config.common import Range, Scale, Crop
from gui.widgets.lazyCanvas.linkedLazyGraphicsItem import LinkedLazyGraphicsItem
from gui.widgets.lazyCanvas.keyLineItem import KeyLineItem

from typing import List, Dict


class SequentialPixmapOverlayCanvasV2(LazyCanvas):
    BACKGROUND_CANVAS_ITEM_Z = 0
    FOREGROUND_CANVAS_ITEM_Z_START = 1
    MAXIMUM_FOREGROUND_CAMVAS_NUM = 10

    DEFAULT_ALPHA = 0.6

    def __init__(self, resourcepool, *args, **kwargs) -> None:
        super().__init__(resourcepool, *args, **kwargs)
        self.bgCanvas: LazySequentialCanvas = None
        self.bgItemQueue = deque([])
        self.fgCanvas: Dict[LazySequentialCanvas] = {}
        self.fgItemQueue: Dict[List] = {}

    def focusAll(self):
        sceneRect = QtCore.QRectF()
        if self.bgCanvas is not None:
            sceneRect = QtCore.QRectF(self.bgCanvas.view.sceneRect())

        for index in self.fgCanvas:
            canvas = self.fgCanvas[index]
            fgSceneRect = canvas.view.sceneRect()

            sceneRect.setLeft(min(sceneRect.left(), fgSceneRect.left()))
            sceneRect.setTop(min(sceneRect.top(), fgSceneRect.top()))
            sceneRect.setRight(max(sceneRect.right(), fgSceneRect.right()))
            sceneRect.setBottom(max(sceneRect.bottom(), fgSceneRect.bottom()))

        self.view.setSceneRect(sceneRect)

    def focusOverlap(self):
        sceneRect = QtCore.QRectF()
        if self.bgCanvas is not None:
            sceneRect = QtCore.QRectF(self.bgCanvas.view.sceneRect())

        for index in self.fgCanvas:
            canvas = self.fgCanvas[index]
            fgSceneRect = canvas.view.sceneRect()

            sceneRect.setLeft(max(sceneRect.left(), fgSceneRect.left()))
            sceneRect.setTop(max(sceneRect.top(), fgSceneRect.top()))
            sceneRect.setRight(min(sceneRect.right(), fgSceneRect.right()))
            sceneRect.setBottom(min(sceneRect.bottom(), fgSceneRect.bottom()))

        self.view.setSceneRect(sceneRect)

    def setBackgroundCanvas(self, background_canvas: LazySequentialCanvas):
        self.bgCanvas = background_canvas

    def detachBackgroundCanvas(self):
        for linked_item in self.bgItemQueue:
            self.view.lazyScene().removeItem(linked_item)
        self.bgItemQueue = deque([])
        self.bgCanvas = None

    def setForegroundCanvas(self, foreground_canvas: LazySequentialCanvas, index: int):
        assert index + 1 > 0, "index should be non-negative"
        if index in self.fgCanvas:
            self.detachForegroundCanvas(index)
        self.fgCanvas[index] = foreground_canvas
        self.fgItemQueue[index] = deque([])

    def detachForegroundCanvas(self, index: int):
        if index not in self.fgCanvas:
            return
        for linked_item in self.fgItemQueue[index]:
            self.view.lazyScene().removeItem(linked_item)
        self.fgItemQueue.pop(index)
        self.fgCanvas.pop(index)

    def addItem(self, item: QtWidgets.QGraphicsItem):
        self.scene.addItem(item)

    def requestRenderOverlay(self):
        self.scene.clear()
        # add background
        item: QtWidgets.QGraphicsItem
        for index in self.fgCanvas:
            canvas: LazySequentialCanvas = self.fgCanvas[index]
            for item in canvas.scene.item_seq:
                linked_item = LinkedLazyGraphicsItem(item)
                linked_item.setZValue(index + 1)
                linked_item.alpha = 0.6
                self.scene.addItem(linked_item)
                linked_item.setPos(item.pos())
                self.fgItemQueue[index].append(linked_item)

        if self.bgCanvas is not None:
            for i, item in enumerate(self.bgCanvas.scene.item_seq):
                linked_item = LinkedLazyGraphicsItem(item)
                linked_item.setZValue(self.BACKGROUND_CANVAS_ITEM_Z)
                self.scene.addItem(linked_item)
                linked_item.setPos(item.pos())
                self.bgItemQueue.append(linked_item)
                linked_item.pixmap_config.horicrop = (
                    self.bgCanvas.view.pixmap_config.horicrop
                )
        self.focusOverlap()

    def adjustForeGroundAlpha(self, index, alpha):
        if len(self.fgItemQueue[index]) > 0:
            for linked_item in self.fgItemQueue[index]:
                linked_item.alpha = alpha

    def bisect(self, item: QtWidgets.QGraphicsItem, queue) -> int:
        return bisect_left(
            SequentialGraphicsScene.ItemList(queue, key=lambda x: x.linked_item.y()),
            item.y(),
        )

    def bisectY(self, y: float, queue) -> int:
        return bisect_left(
            SequentialGraphicsScene.ItemList(queue, key=lambda x: x.linked_item.y()),
            y,
        )

    def resetYAfter(self, index, queue):
        if len(queue) > 0 and index < len(queue):
            y = queue[0].y() if index > 0 and queue[0] else 0
            for i in range(len(queue)):
                if i < index:
                    y += queue[i].size.height()
                else:
                    queue[i].setY(y)
                    y += queue[i].size.height()
        self.view.reload_scene_items()

    def resetOverride(self, queue):
        for i in range(len(queue)):
            queue[i].setEnabled(True)
            queue[i].setVisible(True)
            queue[i].override_scale = QtCore.QPointF(1.0, 1.0)
            queue[i].override_pos_offset = QtCore.QPointF(0.0, 0.0)
            queue[i].updateOverride()

    def resetOverrideAfter(
        self,
        index,
        override_pos_offset,
        override_scale,
        queue,
        disalble_items_before=False,
    ):
        return self.resetOverrideInRange(
            index,
            len(queue),
            override_pos_offset,
            override_scale,
            queue,
            disalble_items_before,
        )

    def resetOverrideInRange(
        self,
        start,
        end,
        override_pos_offset: QtCore.QPointF,
        override_scale: QtCore.QPointF,
        queue,
        disalble_items_before=False,
    ):
        if len(queue) > 0 and start < len(queue) and end <= len(queue):
            y_scale_offset = 0.0
            for i in range(end):
                if i < start:
                    if disalble_items_before:
                        queue[i].setEnabled(False)
                        queue[i].setVisible(False)
                else:
                    queue[i].setEnabled(True)
                    queue[i].setVisible(True)
                    queue[i].override_scale = QtCore.QPointF(override_scale)
                    queue[i].override_pos_offset = QtCore.QPointF(override_pos_offset)
                    queue[i].override_pos_offset.setY(
                        override_pos_offset.y() - y_scale_offset
                    )
                    y_scale_offset += queue[i].linked_item.size.height() * (
                        1.0 - override_scale.y()
                    )

                    queue[i].updateOverride()

        self.view.reload_scene_items()
