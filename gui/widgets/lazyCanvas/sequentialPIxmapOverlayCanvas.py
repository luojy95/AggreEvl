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

from typing import List


class SequentialPixmapOverlayCanvas(LazyCanvas):
    BACKGROUND_CANVAS_ITEM_Z = 0
    FOREGROUND_CANVAS_ITEM_Z = 1

    DEFAULT_ALPHA = 0.6

    def __init__(self, resourcepool, *args, **kwargs) -> None:
        super().__init__(resourcepool, *args, **kwargs)
        self.bgCanvas: LazySequentialCanvas = None
        self.fgCanvas: LazySequentialCanvas = None
        self.fgItemQueue = deque([])
        self.bgItemQueue = deque([])

        self.cached_scales: List[Scale] = []
        self.cached_scales_all: List[Scale] = []
        self.cached_range: Range = Range()
        self.cached_range_all: Range = Range()
        self.cached_offset_x: float = 0
        self.cached_first_item_crop: float = 0
        self.cached_last_item_crop: float = 0

        self.fg_crop: Crop = Crop()

        self.aligned_once = False

        self.cached_display_range = Range()

    def resetSceneRect(self):
        bgSceneRect = QtCore.QRectF()
        if self.bgCanvas is not None:
            bgSceneRect = self.bgCanvas.view.sceneRect()

        fgSceneRect = QtCore.QRectF()
        if self.fgCanvas is not None:
            fgSceneRect = self.fgCanvas.view.sceneRect()

        x = min(bgSceneRect.x(), fgSceneRect.x())
        y = min(bgSceneRect.y(), fgSceneRect.y())
        width = max(bgSceneRect.width(), fgSceneRect.width())
        height = max(bgSceneRect.height(), fgSceneRect.height())

        rect = QtCore.QRectF(x, y, width, height)
        self.view.setSceneRect(rect)

    def setBackgroundCanvas(self, background_canvas: LazySequentialCanvas):
        self.bgCanvas = background_canvas
        self.bgCanvas.view.signals.keypoints_removed.connect(self.alignKeyPointRemoval)

    def detachBackgroundCanvas(self):
        for linked_item in self.bgItemQueue:
            self.view.lazyScene().removeItem(linked_item)
        self.bgItemQueue = deque([])
        self.bgCanvas = None
        self.cached_scales: List[Scale] = []
        self.cached_scales_all: List[Scale] = []
        self.cached_range: Range = Range()
        self.cached_range_all: Range = Range()

    def setForegroundCanvas(self, foreground_canvas: LazySequentialCanvas):
        self.fgCanvas = foreground_canvas
        self.fgCanvas.view.signals.keypoints_removed.connect(self.alignKeyPointRemoval)

    def detachForegroundCanvas(self):
        for linked_item in self.fgItemQueue:
            self.view.lazyScene().removeItem(linked_item)
        self.fgItemQueue = deque([])
        self.fgCanvas = None
        self.fg_crop: Crop = Crop()

    def addItem(self, item: QtWidgets.QGraphicsItem):
        self.scene.addItem(item)

    def requestRenderOverlay(self):
        self.scene.clear()
        # add background
        item: QtWidgets.QGraphicsItem
        if self.fgCanvas is not None:
            for item in self.fgCanvas.scene.item_seq:
                linked_item = LinkedLazyGraphicsItem(item)
                linked_item.setZValue(self.FOREGROUND_CANVAS_ITEM_Z)
                linked_item.alpha = 0.6
                self.scene.addItem(linked_item)
                linked_item.setPos(item.pos())
                self.fgItemQueue.append(linked_item)
                linked_item.item_changed.signal.connect(self.itemChanged)
            self.fg_crop = Crop()

        if self.bgCanvas is not None:
            displayRange = self.bgCanvas.view.view_config.displayRange
            print(displayRange)
            initialScale = self.bgCanvas.view.view_config.initialScale
            for i, item in enumerate(self.bgCanvas.scene.item_seq):
                if displayRange.isValid() and (
                    i < displayRange.start or i >= displayRange.end
                ):
                    continue
                linked_item = LinkedLazyGraphicsItem(item)
                linked_item.setZValue(self.BACKGROUND_CANVAS_ITEM_Z)
                self.scene.addItem(linked_item)
                linked_item.setPos(item.pos())
                self.bgItemQueue.append(linked_item)
                linked_item.pixmap_config.horicrop = (
                    self.bgCanvas.view.pixmap_config.horicrop
                )
            if displayRange.isValid():
                self.cached_range = displayRange
            else:
                self.cached_range = Range(0, len(self.bgCanvas.scene.item_seq))

            self.resetOverrideInRange(
                self.cached_range.start,
                self.cached_range.end,
                QtCore.QPointF(),
                initialScale.toQPointF(),
                self.bgItemQueue,
                True,
            )
            self.cached_scales = [
                deepcopy(initialScale)
                for i in range(self.cached_range.start, self.cached_range.end + 1)
            ]

            self.cached_range_all = deepcopy(self.cached_range)
            self.cached_scales_all = deepcopy(self.cached_scales)
            self.cached_offset_x = 0
            self.cached_last_item_crop = 0
            self.cached_first_item_crop = 0

        self.resetSceneRect()

    def alignKeyPointRemoval(self, values):
        id, center = values
        if self.fgCanvas:
            self.fgCanvas.view.removeKeyPointItemById(id, emit_signal=False)
        if self.bgCanvas:
            self.bgCanvas.view.removeKeyPointItemById(id, emit_signal=False)

    def alignOverlayWithKeyPoints(
        self,
        fg_kps: Dict[int, KeyLineItem],
        bg_kps: Dict[int, KeyLineItem],
    ):
        self.cached_first_item_crop: float = 0
        self.cached_last_item_crop: float = 0
        self.fg_crop = Crop()
        self.cached_range = deepcopy(self.cached_range_all)
        self.cached_scales = deepcopy(self.cached_scales_all)
        self.cached_offset_x = 0
        # sanitize kps
        fg_keys = set(fg_kps.keys())
        bg_keys = set(bg_kps.keys())
        valid_keys = fg_keys.intersection(bg_keys)

        if self.bgCanvas is None or len(valid_keys) <= 0 or len(self.bgItemQueue) <= 0:
            return

        self.resetOverride(self.bgItemQueue)

        sorted_keys = sorted(list(valid_keys), key=lambda k: fg_kps.get(k).center.y())

        prev_y = bg_kps[sorted_keys[0]].center.y()
        for i in range(1, len(sorted_keys)):
            print(prev_y, i, bg_kps[sorted_keys[i]].center.y())
            if prev_y >= bg_kps[sorted_keys[i]].center.y():
                valid_keys.remove(sorted_keys[i])
            else:
                prev_y = bg_kps[sorted_keys[i]].center.y()

        print(f"valid keys: {valid_keys}")

        # get final keys
        final_keys = sorted(list(valid_keys), key=lambda k: fg_kps.get(k).center.y())

        # align first item
        first_key = final_keys[0]
        first_y_bg = bg_kps.get(first_key).center.y()
        first_item_id_bg = self.bgCanvas.scene.bisectY(first_y_bg) - 1
        first_item_id_bg = max(
            0, min(len(self.bgCanvas.scene.item_seq) - 1, first_item_id_bg)
        )

        rate_bg = (
            self.bgItemQueue[0].linked_item.raw_size.width()
            / self.bgItemQueue[0].linked_item.size.width()
        )
        self.cached_first_item_crop: float = (
            first_y_bg - self.bgCanvas.scene.item_seq[first_item_id_bg].y()
        ) * rate_bg
        self.cached_range.start = first_item_id_bg
        first_item_id_bg -= self.cached_range_all.start

        first_y_fg = fg_kps.get(first_key).center.y()
        # first_item_id_fg = self.bisectY(first_y_fg, self.fgItemQueue) - 1
        first_item_id_fg = self.fgCanvas.scene.bisectY(first_y_fg) - 1
        first_item_id_fg = max(
            0, min(len(self.fgCanvas.scene.item_seq) - 1, first_item_id_fg)
        )

        self.cached_display_range.start = first_y_fg

        if len(final_keys) == 1.0:
            override_y = first_y_fg - first_y_bg
            self.resetOverrideAfter(
                first_item_id_bg,
                QtCore.QPointF(0.0, override_y),
                QtCore.QPointF(1.0, self.bgCanvas.view.view_config.initialScale.y),
                self.bgItemQueue,
                disalble_items_before=True,
            )
            self.cached_range.start = first_item_id_bg + self.cached_range_all.start
            self.cached_scales = self.cached_scales[first_item_id_bg:]
            self.cached_display_range.end = (
                self.fgItemQueue[-1].y() + self.fgItemQueue[-1].size.height()
            )
        else:
            last_key = final_keys[-1]
            last_y_bg = bg_kps.get(last_key).center.y()
            last_item_id_bg = self.bgCanvas.scene.bisectY(last_y_bg) - 1
            last_item_id_bg = max(
                0, min(len(self.bgCanvas.scene.item_seq) - 1, last_item_id_bg)
            )
            self.cached_range.end = last_item_id_bg + 1
            self.cached_last_item_crop: float = (
                self.bgCanvas.scene.item_seq[last_item_id_bg].y()
                + self.bgCanvas.scene.item_seq[last_item_id_bg].size.height()
                - last_y_bg
            ) * rate_bg
            print(
                f"cropping: {self.cached_first_item_crop}, {self.cached_last_item_crop}"
            )
            last_item_id_bg -= self.cached_range_all.start
            print(
                "HEIHEI:",
                self.cached_range_all.start,
                first_item_id_bg,
                last_item_id_bg,
            )
            self.cached_scales = self.cached_scales[
                first_item_id_bg : last_item_id_bg + 1
            ]

            last_y_fg = fg_kps.get(last_key).center.y()
            last_item_id_fg = self.fgCanvas.scene.bisectY(last_y_fg) - 1
            last_item_id_fg = max(
                0, min(len(self.fgCanvas.scene.item_seq) - 1, last_item_id_fg)
            )
            self.cached_display_range.end = last_y_fg

            # calculate X regressions
            xs_bg = []
            xs_fg = []
            for key in final_keys:
                xs_bg.append(bg_kps.get(key).center.x())
                xs_fg.append(fg_kps.get(key).center.x())

            X = np.array(xs_bg).reshape(len(final_keys), 1)
            y = np.array(xs_fg)
            reg = LinearRegression().fit(X, y)
            override_x = reg.intercept_
            scale_x = reg.coef_[0]
            for i in range(len(self.cached_scales)):
                self.cached_scales[i].x = scale_x

            rate_fg = (
                self.fgItemQueue[0].linked_item.raw_size.width()
                / self.fgItemQueue[0].linked_item.size.width()
            )

            self.cached_offset_x = override_x * rate_bg

            left_edge_fg = 0
            left_edge_bg = (
                override_x
                + self.bgItemQueue[0].pixmap_config.horicrop.left / rate_bg * scale_x
            )
            if left_edge_bg > left_edge_fg:
                self.fg_crop.left = (left_edge_bg - left_edge_fg) * rate_fg

            right_edge_bg = (
                override_x
                + (
                    self.bgItemQueue[0].linked_item.size.width()
                    - self.bgItemQueue[0].pixmap_config.horicrop.right / rate_bg
                )
                * scale_x
            )
            right_edge_fg = self.fgItemQueue[0].linked_item.size.width()
            if right_edge_fg > right_edge_bg:
                self.fg_crop.right = (right_edge_fg - right_edge_bg) * rate_fg

            for i in range(len(self.fgItemQueue)):
                self.fgItemQueue[i].pixmap_config.horicrop = self.fg_crop
                self.fgItemQueue[i].updateOverride()

            print(rate_fg, right_edge_fg, right_edge_bg, override_x, scale_x)

            print(self.fg_crop)

            left_edge = max(left_edge_fg, left_edge_bg)
            right_edge = min(right_edge_bg, right_edge_fg)

            num_regions = len(final_keys) - 1
            item_start_id_bg = first_item_id_bg
            item_end_id_bg = first_item_id_bg
            last_region_scale_y = None
            offset_ys = []
            for i in range(num_regions):
                start_key = final_keys[i]
                region_start_fg = fg_kps.get(start_key).center.y()
                region_start_bg = bg_kps.get(start_key).center.y()

                start_item_bg: LinkedLazyGraphicsItem = self.bgItemQueue[
                    item_start_id_bg
                ]

                end_key = final_keys[i + 1]
                region_end_fg = fg_kps.get(end_key).center.y()
                region_end_bg = bg_kps.get(end_key).center.y()
                item_end_id_bg = self.bisectY(region_end_bg, self.bgItemQueue) - 1

                if item_start_id_bg == item_end_id_bg:
                    continue

                is_first_frame_in_all = last_region_scale_y is None

                if is_first_frame_in_all:
                    start_offset_y = 0
                else:
                    start_offset_y = (
                        start_item_bg.linked_item.y()
                        + start_item_bg.linked_item.size.height()
                        - region_start_bg
                    )

                end_offset_y = 0
                total_offset = start_offset_y + end_offset_y
                distance_bg = region_end_bg - region_start_bg - total_offset
                if is_first_frame_in_all:
                    distance_fg = region_end_fg - region_start_fg - total_offset
                else:
                    distance_fg = (
                        region_end_fg
                        - region_start_fg
                        - total_offset * last_region_scale_y
                    )
                scale_y = distance_fg / distance_bg

                first_item_height_before_region = (
                    region_start_bg - start_item_bg.linked_item.y()
                )
                first_item_scale = (
                    scale_y if is_first_frame_in_all else last_region_scale_y
                )
                first_item_offset_y = first_item_height_before_region * (
                    1.0 - first_item_scale
                )
                for i in range(
                    item_start_id_bg + (1 - is_first_frame_in_all), item_end_id_bg + 1
                ):
                    self.cached_scales[i - first_item_id_bg].y = scale_y

                offset_y = region_start_fg - region_start_bg
                first_item_scale_offset_y = offset_y + first_item_offset_y

                if is_first_frame_in_all:
                    offset_ys.append(first_item_scale_offset_y)

                following_item_init_offset_y = (
                    first_item_scale_offset_y
                    - start_item_bg.linked_item.size.height() * (1.0 - first_item_scale)
                )
                following_item_heigh_change = 0.0
                print("1. ", following_item_init_offset_y)
                # print(item_start_id_bg, item_end_id_bg)
                # print(offset_ys[-1], first_item_scale_offset_y)
                for i in range(item_start_id_bg + 1, item_end_id_bg + 1):
                    offset_ys.append(
                        following_item_init_offset_y - following_item_heigh_change
                    )
                    following_item_heigh_change += self.bgItemQueue[
                        i
                    ].linked_item.size.height() * (1.0 - scale_y)

                print("2. ", following_item_init_offset_y - following_item_heigh_change)

                # if is_first_frame_in_all:
                print(
                    region_end_fg - region_start_fg,
                    region_end_bg - region_start_bg,
                    scale_y,
                )
                end_item_bg = self.bgItemQueue[item_end_id_bg]
                if is_first_frame_in_all:
                    print(
                        -(
                            start_item_bg.linked_item.y()
                            + (region_start_bg - start_item_bg.linked_item.y())
                            * scale_y
                            + offset_ys[item_start_id_bg - first_item_id_bg]
                            - end_item_bg.linked_item.y()
                            - (region_end_bg - end_item_bg.linked_item.y()) * scale_y
                            - offset_ys[item_end_id_bg - first_item_id_bg]
                        )
                    )
                else:
                    print(
                        -(
                            start_item_bg.linked_item.y()
                            + (region_start_bg - start_item_bg.linked_item.y())
                            * last_region_scale_y
                            + offset_ys[item_start_id_bg - first_item_id_bg]
                            - end_item_bg.linked_item.y()
                            - (region_end_bg - end_item_bg.linked_item.y()) * scale_y
                            - offset_ys[item_end_id_bg - first_item_id_bg]
                        )
                    )

                item_start_id_bg = item_end_id_bg
                last_region_scale_y = scale_y

            print(len(offset_ys), len(self.cached_scales), self.cached_range.size())
            total_height = 0
            for i in range(len(self.bgItemQueue)):
                if (
                    i < self.cached_range.start - self.cached_range_all.start
                    or i >= self.cached_range.end - self.cached_range_all.start
                ):
                    self.bgItemQueue[i].setEnabled(False)
                    self.bgItemQueue[i].setVisible(False)
                else:
                    self.bgItemQueue[i].setEnabled(True)
                    self.bgItemQueue[i].setVisible(True)
                    self.bgItemQueue[i].override_scale = QtCore.QPointF(
                        self.cached_scales[i - first_item_id_bg].x,
                        self.cached_scales[i - first_item_id_bg].y,
                    )
                    self.bgItemQueue[i].override_pos_offset = QtCore.QPointF(
                        override_x, offset_ys[i - first_item_id_bg]
                    )
                    total_height += (
                        self.bgItemQueue[i].linked_item.size.height()
                        * self.cached_scales[i - first_item_id_bg].y
                    )
                    self.bgItemQueue[i].updateOverride()
            self.aligned_once = True

            print(
                f"alignment: {first_y_fg}, {last_y_fg}, {last_y_fg - first_y_fg}, {total_height}"
            )

            self.view.setSceneRect(
                left_edge, first_y_fg, right_edge - left_edge, last_y_fg - first_y_fg
            )

    def adjustForeGroundAlpha(self, alpha):
        if len(self.fgItemQueue) > 0:
            for linked_item in self.fgItemQueue:
                linked_item.alpha = alpha

    def itemChanged(self, item: QtWidgets.QGraphicsItem):
        if item.zValue() == self.FOREGROUND_CANVAS_ITEM_Z:
            index = self.bisect(item, self.fgItemQueue)
            self.resetYAfter(index, self.fgItemQueue)

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
