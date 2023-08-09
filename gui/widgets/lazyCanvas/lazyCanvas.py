from bisect import bisect_left
from collections import deque

from PySide6.QtWidgets import (
    QWidget,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QHBoxLayout,
    QGraphicsRectItem,
    QGraphicsLineItem,
)
from PySide6.QtCore import Qt, QLine
from PySide6.QtGui import (
    QPaintEvent,
    QCloseEvent,
    QKeyEvent,
    QWheelEvent,
    QColor,
    QMouseEvent,
    QBrush,
    QPen,
)
from PySide6.QtCore import QRectF, QSize, QObject, Signal, QPointF

from gui.widgets.lazyCanvas.lazyGraphicsItem import LazyGraphicsPixmapItem
from gui.widgets.lazyCanvas.config.lazyGraphicsPixmapItemConfig import (
    LazyGraphicsPixmapItemConfig,
)
from gui.widgets.lazyCanvas.config.lazyCanvasViewConfig import LazyCanvasViewConfig
from gui.widgets.lazyCanvas.config.common import Range, Scale, Crop
from gui.widgets.lazyCanvas.common import *
from gui.widgets.lazyCanvas.keyLineItem import KeyLineItem
from gui.widgets.lazyCanvas.keyPointItem import KeyPointItem
from gui.widgets.taggedQGraphicsRect import TaggedQGraphicsRectItem
from gui.resourcepool import ResourcePool


class LazyGraphicsViewSignals(QObject):
    keypoints_removed: Signal = Signal(object)
    keypoints_set: Signal = Signal(object)
    calibration_ball_set: Signal = Signal(object)
    wheel_event: Signal = Signal(object)
    key_down_event: Signal = Signal(object)
    key_up_event: Signal = Signal(object)
    analysis_rect_updated: Signal = Signal(object)


class LazyGraphicsView(QGraphicsView):
    def __init__(self, resourcepool, scene: "LazyGraphicsScene", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # bind scene
        self.setScene(scene)
        if isinstance(scene, SequentialGraphicsScene):
            scene.view_scene_connect.request_paint_signal.connect(
                self.reload_scene_items
            )

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.centerOn(0, 0)

        self.current_visible_region = QRectF()
        self.current_visible_items = set()

        self.resourcepool: ResourcePool = resourcepool
        self.pressed_keys: set[int] = set()

        # zooming
        self.zoom_factor: float = 1.25
        self.current_zoom: float = 1.0

        # any rangy selection
        self.select_mode = False
        self.prev_mouse_pos = None
        self.select_rect_item = None  # selection aux tool

        # resizing
        self.ready_to_resize_target = None
        self.is_resizing = False
        self.virtual_resize_item = None  # resizing aux item

        self.view_config = LazyCanvasViewConfig()
        self.pixmap_config: LazyGraphicsPixmapItemConfig = (
            LazyGraphicsPixmapItemConfig()
        )

        # TODO: clean up auto-grouping
        self.enable_auto_grouping = True

        # horizontal cropping aux items
        self.crop_rect_left = None
        self.crop_rect_right = None

        # key point selection aux items
        self.kp_map = dict()
        self.current_kp_id = 0
        self.kp_virtual = None

        # interactive options
        self.is_zoom_by_scroll_enabled = True
        self.is_item_select_enabled = False
        self.is_key_point_mode_enabled = False
        self.is_vsplit_mode_enabled = False
        self.is_any_range_select_enabled = False
        self.is_item_deletion_enabled = False
        self.is_item_resize_enabled = False
        self.is_hover_show_analysis_rect_enabled = False

        self.is_calibration_ball_mode_enabled = False
        self.is_drawing_calibration_ball = False
        self.calibration_ball_rect = None
        self.calibration_ball_diameter_pixel = None

        self.analysis_rect = None
        self.analysis_rect_height = None
        self.analysis_rect_dist_to_center = None

        # signals
        self.signals = LazyGraphicsViewSignals()

        self.crop_rect_alpha = 128

    def enableKeyPointMode(self, enable):
        if self.is_key_point_mode_enabled != enable:
            self.is_key_point_mode_enabled = enable
            if enable:
                for item in self.current_visible_items:
                    item.setSelected(False)
            else:
                if self.kp_virtual:
                    self.lazyScene().removeAuxItem(self.kp_virtual)
                    self.kp_virtual = None

    def enableAutoGrouping(self, enable):
        # TODO: clean up
        pass
        # scene = self.lazyScene()
        # if self.enable_auto_grouping != enable and isinstance(
        #     scene, SequentialGraphicsScene
        # ):
        #     self.enable_auto_grouping = enable
        #     scene.enable_auto_grouping = enable
        #     self.reload_scene_items()

    def auto_grouping(self):
        # TODO: clean up
        pass
        # scene = self.lazyScene()
        # if self.enable_auto_grouping and isinstance(scene, SequentialGraphicsScene):
        #     scene.autoGrouping(self.size(), self.current_visible_region)

    def lazyScene(self) -> "LazyGraphicsScene":
        if isinstance(self.scene(), LazyGraphicsScene):
            return self.scene()
        else:
            assert False, "didn't initialize lazyGraphicsView with lazyGraphicsScene"
            return None

    def initSceneRect(self):
        if isinstance(self.scene(), SequentialGraphicsScene):
            items = self.scene().item_seq
            num_items = len(items)

            max_width = 0
            max_height = 0
            if num_items > 0:
                first_item = items[0]
                max_width = first_item.size.width()
                max_height = items[-1].size.height() + items[-1].y()
            self.setSceneRect(0, 0, max_width, max_height)
            self.scale(1.0 / self.current_zoom, 1.0 / self.current_zoom)
            self.setup_initial_scaling()

    def setup_initial_scaling(self):
        if isinstance(self.scene(), SequentialGraphicsScene):
            items = self.scene().item_seq
            num_items = len(items)

            max_width = 0
            if num_items > 0:
                first_item = items[0]
                max_width = first_item.size.width()

            current_width = self.size().width()

            if current_width > 0 and max_width > 0:
                self.current_zoom = current_width / max_width * 0.9

                self.scale(self.current_zoom, self.current_zoom)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        self.setBackgroundBrush(QColor("black"))
        self.reload_scene_items()

    def update_crop_rect(self):
        if isinstance(self.scene(), SequentialGraphicsScene):
            items = self.scene().item_seq
            num_items = len(items)

            if num_items > 0:
                first_item = items[0]

                rate = first_item.size.width() * 1.0 / first_item.raw_size.width()

                if self.pixmap_config is None:
                    return
                crop_left, crop_right = (
                    round(rate * self.pixmap_config.horicrop.left),
                    round(rate * self.pixmap_config.horicrop.right),
                )
                if self.crop_rect_left is not None:
                    self.lazyScene().removeAuxItem(self.crop_rect_left)
                if self.crop_rect_right is not None:
                    self.lazyScene().removeAuxItem(self.crop_rect_right)

                left_rect = QRectF(
                    0, self.sceneRect().y(), crop_left, self.sceneRect().height()
                )
                right_rect = QRectF(
                    self.sceneRect().width() - crop_right,
                    self.sceneRect().y(),
                    crop_right,
                    self.sceneRect().height(),
                )

                self.crop_rect_left = QGraphicsRectItem(left_rect)
                self.crop_rect_right = QGraphicsRectItem(right_rect)
                self.crop_rect_left.setZValue(UPMOST_LAYER_Z)
                self.crop_rect_right.setZValue(UPMOST_LAYER_Z)

                pen = QPen()
                pen.setColor(QColor(0, 0, 0, self.crop_rect_alpha))
                pen.setCapStyle(Qt.RoundCap)
                self.crop_rect_left.setBrush(QColor(0, 0, 0, self.crop_rect_alpha))
                self.crop_rect_left.setPen(pen)
                self.lazyScene().addAuxItem(self.crop_rect_left)

                self.crop_rect_right.setBrush(QColor(0, 0, 0, self.crop_rect_alpha))
                self.crop_rect_right.setPen(pen)
                self.lazyScene().addAuxItem(self.crop_rect_right)

    def update_scale_y(self, sy):
        if self.view_config is None:
            self.view_config = LazyCanvasViewConfig()

        self.scale(1.0, 1.0 / self.view_config.initialScale.y)
        self.scale(1.0, sy)

        self.view_config.initialScale.y = sy

    def update_pixmap_config(self, config: LazyGraphicsPixmapItemConfig):
        if config != self.pixmap_config:
            self.pixmap_config = config
            self.update_scene_items_config()
            self.update_crop_rect()

    def update_scene_items_config(self):
        for item in self.scene().items():
            if (
                isinstance(item, LazyGraphicsPixmapItem)
                and item.zValue() < UPMOST_LAYER_Z
            ):
                item.update_config(self.pixmap_config)
                # item.preview_config()

        for i, item in enumerate(self.current_visible_items):
            if isinstance(item, LazyGraphicsPixmapItem):
                item.preview_config()

    def reload_scene_items(self):
        # self.auto_grouping()
        visible_region = self.mapToScene(self.viewport().rect()).boundingRect()
        visible_items = set(self.scene().items(visible_region))
        if (
            visible_region != self.current_visible_region
            or visible_items != self.current_visible_items
        ):
            appearing_items = visible_items.difference(self.current_visible_items)
            for i, item in enumerate(appearing_items):
                if isinstance(item, LazyGraphicsPixmapItem):
                    item.load(self.resourcepool)
                    item.setFlag(
                        QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                        self.is_item_select_enabled,
                    )

            disappering_items = self.current_visible_items.difference(visible_items)
            for i, item in enumerate(disappering_items):
                if isinstance(item, LazyGraphicsPixmapItem):
                    item.release(self.resourcepool)
                    item.setFlag(
                        QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                        False,
                    )

            self.current_visible_region = visible_region
            self.current_visible_items = visible_items

    def fresh_visible_items(self):
        for i, item in enumerate(self.current_visible_items):
            if isinstance(item, LazyGraphicsPixmapItem):
                item.load(self.resourcepool)
                item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                    self.is_item_select_enabled,
                )

    def keyPressEvent(self, event: QKeyEvent, emit=True) -> None:
        if emit:
            self.signals.key_down_event.emit((self, event))
        if self.is_item_deletion_enabled and event.key() == Qt.Key.Key_Delete:
            for item in self.scene().selectedItems():
                self.scene().removeItem(item)
        else:
            self.pressed_keys.add(event.key())
            self.config_cursors(added_key=event.key())
            return super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent, emit=True) -> None:
        if emit:
            self.signals.key_up_event.emit((self, event))
        if event.key() in self.pressed_keys:
            self.pressed_keys.remove(event.key())
            self.config_cursors(removed_key=event.key())

        return super().keyReleaseEvent(event)

    def config_cursors(self, added_key=None, removed_key=None):
        if added_key is not None:
            if self.is_vsplit_mode_enabled and Qt.Key.Key_Alt == added_key:
                QWidget.setCursor(self, Qt.SplitVCursor)
            elif self.is_any_range_select_enabled and Qt.Key.Key_Shift == added_key:
                QWidget.setCursor(self, Qt.CrossCursor)
            elif self.is_item_resize_enabled and Qt.Key.Key_Control == added_key:
                QWidget.setCursor(self, Qt.SizeBDiagCursor)

        if removed_key is not None:
            QWidget.setCursor(self, Qt.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent, emit=True) -> None:
        if emit:
            self.signals.wheel_event.emit((self, event))
        if self.is_zoom_by_scroll_enabled and Qt.Key.Key_Control in self.pressed_keys:
            is_zoom_in = event.angleDelta().y() > 0
            zoom_factor = self.zoom_factor if is_zoom_in else 1.0 / self.zoom_factor
            self.current_zoom *= zoom_factor

            self.scale(zoom_factor, zoom_factor)

            # TODO: fix zooming with viewport followed
            # prev_position = self.mapToScene(event.position().toPoint())
            # delta = self.mapToScene(event.position().toPoint()) - prev_position
            # self.translate(delta.x(), delta.y())
        else:
            return super().wheelEvent(event)

    def removeKeyPointItem(self, item: KeyPointItem, emit_signal=True):
        return self.removeKeyPointItemById(item.id, emit_signal)

    def removeKeyPointItemById(self, item_id: int, emit_signal=True):
        if item_id in self.kp_map:
            if item_id == self.current_kp_id - 1:
                self.current_kp_id = item_id
            item = self.kp_map.pop(item_id)
            self.lazyScene().removeAuxItem(item)
            if emit_signal:
                self.signals.keypoints_removed.emit((item_id, item.center))

    def update_analysis_rect(self):
        viewport_center = self.mapToScene(self.viewport().rect().center())
        viewport_topleft = self.mapToScene(self.viewport().rect().topLeft())

        if self.analysis_rect is not None:
            self.lazyScene().removeAuxItem(self.analysis_rect)
            self.analysis_rect = None

        if (
            self.analysis_rect_height is not None
            and self.analysis_rect_dist_to_center is not None
        ):
            rect = QRectF(
                viewport_topleft.x(),
                viewport_center.y() + self.analysis_rect_dist_to_center,
                self.scene().width(),
                self.analysis_rect_height,
            )

            self.analysis_rect_dist_to_center = rect.y() - viewport_center.y()

            self.analysis_rect = QGraphicsRectItem(rect)
            pen = QPen()
            pen.setColor(QColor(247, 127, 0, 128))
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidth(3)
            self.analysis_rect.setBrush(QColor(247, 127, 0, 128))
            self.analysis_rect.setPen(pen)
            self.analysis_rect.setZValue(UPMOST_LAYER_Z)
            self.lazyScene().addAuxItem(self.analysis_rect)
            self.signals.analysis_rect_updated.emit(rect.center().y())
        else:
            self.signals.analysis_rect_updated.emit(None)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_hover_show_analysis_rect_enabled:
            key_point = self.mapToScene(event.position().toPoint())
            viewport_center = self.mapToScene(self.viewport().rect().center())
            viewport_topleft = self.mapToScene(self.viewport().rect().topLeft())
            viewport_bottomright = self.mapToScene(self.viewport().rect().bottomRight())
            x_out = (
                key_point.x() <= viewport_topleft.x() + 20
                or key_point.x() >= viewport_bottomright.x() - 20
            )
            y_out = (
                key_point.y() <= viewport_topleft.y() + 20
                or key_point.y() >= viewport_bottomright.y() - 20
            )
            if self.analysis_rect is not None:
                self.lazyScene().removeAuxItem(self.analysis_rect)
                self.analysis_rect = None
                self.analysis_rect_dist_to_center = None

            if self.analysis_rect_height is not None and not (x_out or y_out):
                rect = QRectF(
                    viewport_topleft.x(),
                    key_point.y() - self.analysis_rect_height / 2,
                    self.scene().width(),
                    self.analysis_rect_height,
                )

                self.analysis_rect_dist_to_center = rect.y() - viewport_center.y()

                self.analysis_rect = QGraphicsRectItem(rect)
                pen = QPen()
                pen.setColor(QColor(247, 127, 0, 128))
                pen.setCapStyle(Qt.RoundCap)
                pen.setWidth(3)
                self.analysis_rect.setBrush(QColor(247, 127, 0, 128))
                self.analysis_rect.setPen(pen)
                self.analysis_rect.setZValue(UPMOST_LAYER_Z)
                self.lazyScene().addAuxItem(self.analysis_rect)
                self.signals.analysis_rect_updated.emit(rect.center().y())
            else:
                self.signals.analysis_rect_updated.emit(None)

        elif self.is_key_point_mode_enabled:
            # xmin = 0
            # xmax = max(1, self.sceneRect().width() - 3)

            # y = self.mapToScene(event.position().toPoint()).y()
            # line = QLine(xmin, y, xmax, y)
            key_point = self.mapToScene(event.position().toPoint())

            if self.kp_virtual is not None:
                self.lazyScene().removeAuxItem(self.kp_virtual)

            self.kp_virtual = KeyPointItem(key_point, self.current_kp_id)
            pen = QPen()
            pen.setColor("orange")
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidth(3)
            self.kp_virtual.setBrush(QColor(255, 128, 0, 255))
            self.kp_virtual.setPen(pen)
            self.kp_virtual.setZValue(UPMOST_LAYER_Z)
            self.lazyScene().addAuxItem(self.kp_virtual)

        elif self.is_calibration_ball_mode_enabled and self.is_drawing_calibration_ball:
            if self.prev_mouse_pos is None:
                self.prev_mouse_pos = self.mapToScene(event.position().toPoint())
            else:
                cur_mouse_pos = self.mapToScene(event.position().toPoint())
                left_x = min(self.prev_mouse_pos.x(), cur_mouse_pos.x())
                right_x = max(self.prev_mouse_pos.x(), cur_mouse_pos.x())
                top_y = min(self.prev_mouse_pos.y(), cur_mouse_pos.y())
                bottom_y = max(self.prev_mouse_pos.y(), cur_mouse_pos.y())
                calibration_rect = QRectF(
                    left_x, top_y, right_x - left_x, bottom_y - top_y
                )
                if calibration_rect.height() <= 0 or calibration_rect.width() <= 0:
                    return
                if self.calibration_ball_rect is not None:
                    self.lazyScene().removeAuxItem(self.calibration_ball_rect)
                    self.calibration_ball_diameter_pixel = None

                ball_string = "calibration ball"
                if (
                    isinstance(self.lazyScene(), SequentialGraphicsScene)
                    and len(self.lazyScene().item_seq) > 0
                ):
                    first_item: LazyGraphicsPixmapItem = self.lazyScene().item_seq[0]
                    ratio = first_item.raw_size.width() / first_item.size.width()
                    ball_diameter = calibration_rect.width() * ratio
                    ball_string = f"calibration ball: D = {round(ball_diameter)} pixel"
                self.calibration_ball_diameter_pixel = ball_diameter
                self.calibration_ball_rect = TaggedQGraphicsRectItem(
                    calibration_rect, ball_string
                )
                pen = QPen()
                pen.setColor("red")
                pen.setCapStyle(Qt.RoundCap)
                pen.setWidth(2)
                self.calibration_ball_rect.setPen(pen)
                self.calibration_ball_rect.setZValue(UPMOST_LAYER_Z)
                self.lazyScene().addAuxItem(self.calibration_ball_rect)

        elif (
            self.is_any_range_select_enabled
            and Qt.Key.Key_Shift in self.pressed_keys
            and self.select_mode
        ):
            if self.prev_mouse_pos is None:
                self.prev_mouse_pos = self.mapToScene(event.position().toPoint())
            else:
                cur_mouse_pos = self.mapToScene(event.position().toPoint())

                left_x = min(self.prev_mouse_pos.x(), cur_mouse_pos.x())
                right_x = max(self.prev_mouse_pos.x(), cur_mouse_pos.x())
                top_y = min(self.prev_mouse_pos.y(), cur_mouse_pos.y())
                bottom_y = max(self.prev_mouse_pos.y(), cur_mouse_pos.y())

                if isinstance(self.lazyScene(), SequentialGraphicsScene):
                    if len(self.lazyScene().item_seq) > 0:
                        left_x = self.lazyScene().item_seq[0].x()
                        right_x = left_x + self.lazyScene().item_seq[0].size.width()

                selection_rect = QRectF(
                    left_x, top_y, right_x - left_x, bottom_y - top_y
                )

                if selection_rect.height() <= 0 or selection_rect.width() <= 0:
                    return

                if self.select_rect_item is not None:
                    self.lazyScene().deselectItemInRect(self.select_rect_item.rect())
                    self.lazyScene().removeAuxItem(self.select_rect_item)

                self.select_rect_item = QGraphicsRectItem(selection_rect)
                self.select_rect_item.setZValue(UPMOST_LAYER_Z)
                pen = QPen()
                pen.setColor(Qt.green)
                pen.setCapStyle(Qt.RoundCap)
                # pen.setDashPattern(Qt.DashLine)
                pen.setWidth(1)

                self.lazyScene().selectItemInRect(selection_rect)

                self.select_rect_item.setBrush(QColor(0, 255, 0, 100))
                self.select_rect_item.setPen(pen)
                self.lazyScene().addAuxItem(self.select_rect_item)

        elif self.is_item_resize_enabled and Qt.Key.Key_Control in self.pressed_keys:
            mouse_pos = event.position().toPoint()

            if not self.is_resizing:
                item = self.itemAt(mouse_pos)

                shoud_switch_to_item_resize = False
                if (
                    item is not None
                    and item.isSelected()
                    and isinstance(item, LazyGraphicsPixmapItem)
                ):
                    mouse_scene_pos = self.mapToScene(mouse_pos)
                    threshold = 5
                    shoud_switch_to_item_resize = (
                        -threshold
                        < mouse_scene_pos.y()
                        - (item.scenePos().toPoint().y() + item.size.height())
                        < threshold
                    )

                if shoud_switch_to_item_resize:
                    QWidget.setCursor(self, Qt.SizeVerCursor)
                    self.ready_to_resize_target = item
                else:
                    QWidget.setCursor(self, Qt.SizeBDiagCursor)
                    self.ready_to_resize_target = None
            else:
                cur_mouse_pos = self.mapToScene(event.position().toPoint())
                self.ready_to_resize_target: LazyGraphicsPixmapItem
                left_x = self.ready_to_resize_target.x()
                right_x = (
                    self.ready_to_resize_target.x()
                    + self.ready_to_resize_target.size.width()
                )
                top_y = self.ready_to_resize_target.y()
                bottom_y = max(
                    self.ready_to_resize_target.y(),
                    cur_mouse_pos.y(),
                )

                virtual_rect = QRectF(left_x, top_y, right_x - left_x, bottom_y - top_y)

                if virtual_rect.height() <= 0 or virtual_rect.width() <= 0:
                    return

                if self.virtual_resize_item is not None:
                    self.lazyScene().removeAuxItem(self.virtual_resize_item)

                self.virtual_resize_item = QGraphicsRectItem(virtual_rect)
                self.virtual_resize_item.setZValue(UPMOST_LAYER_Z)
                pen = QPen()
                pen.setColor(Qt.yellow)
                pen.setCapStyle(Qt.RoundCap)
                pen.setWidth(1)

                self.virtual_resize_item.setBrush(QColor(255, 255, 0, 100))
                self.virtual_resize_item.setPen(pen)
                self.lazyScene().addAuxItem(self.virtual_resize_item)

        else:
            super().mouseMoveEvent(event)

    def overwriteKpPointsWithoutEmit(self, kpmap):
        self.cleanupKeyPoints()
        self.current_kp_id = -1
        for key in kpmap:
            self.current_kp_id = max(self.current_kp_id, key + 1)
            kp = KeyPointItem(QPointF(kpmap[key].x, kpmap[key].y), key)
            kp.isRemoving.signal.connect(self.removeKeyPointItem)

            pen = QPen()
            pen.setColor(QColor(0, 255, 0, 255))
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidth(3)
            kp.setBrush(QColor(0, 255, 0, 255))
            kp.setPen(pen)
            kp.setZValue(UPMOST_LAYER_Z)
            self.lazyScene().addAuxItem(kp)
            self.kp_map[key] = kp

    def overwriteCalibrationBallWithoutEmit(self, ball):
        self.cleanupCalibrationBall()
        calibration_rect = QRectF(ball.x, ball.y, ball.w, ball.h)
        ball_string = "calibration ball"
        if (
            isinstance(self.lazyScene(), SequentialGraphicsScene)
            and len(self.lazyScene().item_seq) > 0
        ):
            first_item: LazyGraphicsPixmapItem = self.lazyScene().item_seq[0]
            ratio = first_item.raw_size.width() / first_item.size.width()
            ball_diameter = calibration_rect.width() * ratio
            ball_string = f"calibration ball: D = {round(ball_diameter)} pixel"
        self.calibration_ball_diameter_pixel = ball_diameter
        self.calibration_ball_rect = TaggedQGraphicsRectItem(
            calibration_rect, ball_string
        )
        pen = QPen()
        pen.setColor("red")
        pen.setCapStyle(Qt.RoundCap)
        pen.setWidth(2)
        self.calibration_ball_rect.setPen(pen)
        self.calibration_ball_rect.setZValue(UPMOST_LAYER_Z)
        self.lazyScene().addAuxItem(self.calibration_ball_rect)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.select_mode = False
        self.prev_mouse_pos = None
        if self.is_drawing_calibration_ball:
            self.is_drawing_calibration_ball = False
            if self.calibration_ball_rect != None:
                self.signals.calibration_ball_set.emit(
                    self.calibration_ball_rect.rect()
                )
        if self.select_rect_item is not None:
            self.lazyScene().removeAuxItem(self.select_rect_item)
            if isinstance(self.lazyScene(), SequentialGraphicsScene):
                seq_scene: SequentialGraphicsScene = self.lazyScene()
                seq_scene.splitAndMergeInRect(self.select_rect_item.rect())
            self.select_rect_item = None

        if (
            self.is_resizing
            and self.ready_to_resize_target is not None
            and self.virtual_resize_item is not None
        ):
            new_size = self.virtual_resize_item.rect().size().toSize()
            self.ready_to_resize_target.resize(new_size, self.resourcepool)
            self.lazyScene().removeAuxItem(self.virtual_resize_item)
            if isinstance(self.lazyScene(), SequentialGraphicsScene):
                seq_scene: SequentialGraphicsScene = self.lazyScene()
                seq_scene.setupSeqAfterItemChanged(self.ready_to_resize_target)
        self.virtual_resize_item = None
        self.ready_to_resize_target = None
        self.is_resizing = False

        return super().mouseReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            if self.is_key_point_mode_enabled and self.kp_virtual:
                self.kp_map[self.current_kp_id] = self.kp_virtual
                self.kp_virtual = None
                self.kp_map[self.current_kp_id].isRemoving.signal.connect(
                    self.removeKeyPointItem
                )
                pen = QPen()
                pen.setColor(Qt.green)
                pen.setCapStyle(Qt.RoundCap)
                pen.setWidth(3)
                self.kp_map[self.current_kp_id].setBrush(QColor(0, 255, 0, 255))
                self.kp_map[self.current_kp_id].setPen(pen)
                self.signals.keypoints_set.emit(
                    (self.current_kp_id, self.kp_map[self.current_kp_id].center)
                )
                self.current_kp_id += 1

            elif self.is_calibration_ball_mode_enabled:
                if self.calibration_ball_rect is not None:
                    self.lazyScene().removeAuxItem(self.calibration_ball_rect)
                    self.calibration_ball_rect = None
                    self.calibration_ball_diameter_pixel = None
                self.is_drawing_calibration_ball = True

            elif self.is_vsplit_mode_enabled and Qt.Key.Key_Alt in self.pressed_keys:
                split_pos = event.position().toPoint()
                item = self.itemAt(split_pos)
                if (
                    item is not None
                    and item.isSelected()
                    and isinstance(item, LazyGraphicsPixmapItem)
                ):
                    item_pos = item.scenePos().toPoint()
                    split_pos_scene = self.mapToScene(split_pos)
                    split = QPointF(0, split_pos_scene.y() - item_pos.y())
                    item.vsplit(split, self.resourcepool)
            elif (
                self.is_any_range_select_enabled
                and Qt.Key.Key_Shift in self.pressed_keys
            ):
                self.select_mode = True
            elif (
                self.is_item_resize_enabled and Qt.Key.Key_Control in self.pressed_keys
            ):
                if (
                    self.ready_to_resize_target is not None
                    and self.ready_to_resize_target in self.scene().items()
                ):
                    self.is_resizing = True
            else:
                return super().mousePressEvent(event)

    def cleanupKeyPoints(self):
        if self.kp_virtual and self.kp_virtual.scene() == self.scene():
            self.lazyScene().removeAuxItem(self.kp_virtual)

        for key in self.kp_map:
            if self.kp_map[key] and self.kp_map[key].scene() == self.scene():
                self.lazyScene().removeAuxItem(self.kp_map[key])

        self.kp_map.clear()

        self.current_kp_id = 0

    def cleanupCalibrationBall(self):
        if (
            self.calibration_ball_rect
            and self.calibration_ball_rect.scene() == self.scene()
        ):
            self.lazyScene().removeAuxItem(self.calibration_ball_rect)
            self.calibration_ball_rect = None
            self.calibration_ball_diameter_pixel = None

    def cleanupAuxItem(self):
        if self.crop_rect_left and self.crop_rect_left.scene() == self.scene():
            self.lazyScene().removeAuxItem(self.crop_rect_left)
            self.crop_rect_left = None

        if self.crop_rect_right and self.crop_rect_right.scene() == self.scene():
            self.lazyScene().removeAuxItem(self.crop_rect_right)
            self.crop_rect_right = None

        self.cleanupKeyPoints()

        self.cleanupCalibrationBall()

        if self.analysis_rect and self.analysis_rect.scene() == self.scene():
            self.lazyScene().removeAuxItem(self.analysis_rect)
            self.analysis_rect_dist_to_center = None
            self.analysis_rect_height = None
            self.analysis_rect = None

    def cleanupConfigs(self):
        self.view_config = LazyCanvasViewConfig()
        self.pixmap_config = LazyGraphicsPixmapItemConfig()


class LazyCanvas(QWidget):
    def __init__(self, resourcepool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLayout(QHBoxLayout())
        self.scene: LazyGraphicsScene = LazyGraphicsScene(resourcepool)
        self.view: LazyGraphicsView = LazyGraphicsView(resourcepool, self.scene, self)
        self.layout().addWidget(self.view)

    def addItem(self, item: QGraphicsItem):
        self.scene.addItem(item)

    def clearSceneItems(self):
        self.view.cleanupAuxItem()
        num_items = len(self.scene.items())
        for _ in range(num_items):
            item = self.scene.items()[0]
            if isinstance(item, LazyGraphicsPixmapItem):
                item.release(self.view.resourcepool)
            self.scene.removeItem(item)
            if isinstance(item, LazyGraphicsPixmapItem):
                item.clean_tempfile()
        self.view.current_visible_items = set()
        self.view.current_visible_region = QRectF()
        self.view.current_zoom = 1.0

    def set_pixmap_config(self, config: LazyGraphicsPixmapItemConfig):
        self.view.update_pixmap_config(config)


class ViewSceneConnect(QObject):
    request_paint_signal = Signal(arguments=None)


class LazyGraphicsScene(QGraphicsScene):
    def __init__(self, resourcepool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resourcepool = resourcepool
        self.view_scene_connect = ViewSceneConnect()

    def addAuxItem(self, item: QGraphicsItem) -> None:
        return super().addItem(item)

    def removeAuxItem(self, item: QGraphicsItem) -> None:
        return super().removeItem(item)

    def selectItemInRect(self, rect: QRectF):
        for item in self.items(rect):
            item.setSelected(True)

    def deselectItemInRect(self, rect: QRectF):
        for item in self.items(rect):
            item.setSelected(False)


class SequentialGraphicsScene(LazyGraphicsScene):
    class ItemList(object):
        # bisect doesn't accept a key function before 3.10,
        # so we build the key into our sequence.
        def __init__(self, l, key):
            self.l = l
            self.key = key

        def __len__(self):
            return len(self.l)

        def __getitem__(self, index):
            return self.key(self.l[index])

    def __init__(self, resourcepool, *args, **kwargs):
        super().__init__(resourcepool, *args, **kwargs)
        self.item_seq = deque()
        self.enable_auto_grouping = False
        self.last_auto_grouping_settings = None

    def addItem(self, item: QGraphicsItem) -> None:
        super().addItem(item)
        index = self.bisect(item)
        self.item_seq.insert(index, item)
        self.resetYAfter(index)

    def addAuxItem(self, item: QGraphicsItem) -> None:
        super().addItem(item)

    def removeItem(self, item: QGraphicsItem) -> None:
        super().removeItem(item)
        index = self.bisect(item)
        self.item_seq.remove(item)
        self.resetYAfter(index)

    def setupSeqAfterItemChanged(self, item: QGraphicsItem) -> None:
        index = self.bisect(item)
        self.resetYAfter(index)

    def removeAuxItem(self, item: QGraphicsItem) -> None:
        super().removeItem(item)

    def autoGrouping(self, view_size: QSize, region: QRectF):
        pass
        # TODO: cleanup
        # if self.last_auto_grouping_settings != (view_size, region):
        #     self.last_auto_grouping_settings = (view_size, region)

        #     view_height = view_size.height()
        #     region_height = region.height()
        #     if view_height == 0 or region_height == 0:
        #         return

        #     current_pixel_scale = view_height * 1.0 / region_height
        #     minimum_height_on_screen = min(100, view_height)
        #     minimum_pixels_group = minimum_height_on_screen / current_pixel_scale

        #     start_i = self.bisectY(region.top())
        #     end_i = self.bisectY(region.bottom())

        #     if start_i <= 0 or end_i <= 0:
        #         return

        #     # split the first & last item
        #     first_item_id = start_i - 1
        #     first_item = self.item_seq[first_item_id]

        #     if not isinstance(first_item, LazyGraphicsPixmapItem):
        #         return

        #     last_item_id = min(len(self.item_seq), end_i) - 1

        #     start_id_for_merge = -1
        #     end_id_for_merge = -1
        #     pixels_for_merge = -1

        #     merge_tasks = deque([])
        #     for i in range(first_item_id, last_item_id + 1):
        #         qimg = self.item_seq[i].qimage
        #         if qimg is None:
        #             if start_id_for_merge != -1:
        #                 merge_tasks.appendleft((start_id_for_merge, end_id_for_merge))
        #                 start_id_for_merge = end_id_for_merge = pixels_for_merge = -1

        #             start_id_for_merge = i
        #             end_id_for_merge = i + 1
        #             pixels_for_merge = minimum_pixels_group
        #             continue

        #         item_height = self.item_seq[i].size.height()
        #         if start_id_for_merge == -1:  # create new task
        #             start_id_for_merge = i
        #             end_id_for_merge = i + 1
        #             pixels_for_merge = item_height
        #         else:
        #             end_id_for_merge = i + 1
        #             pixels_for_merge += item_height

        #         if pixels_for_merge >= minimum_pixels_group:
        #             merge_tasks.appendleft((start_id_for_merge, end_id_for_merge))
        #             start_id_for_merge = end_id_for_merge = pixels_for_merge = -1

        #     if start_id_for_merge != -1:
        #         merge_tasks.appendleft((start_id_for_merge, end_id_for_merge))
        #         start_id_for_merge = end_id_for_merge = pixels_for_merge = -1

        #     for start, end in merge_tasks:
        #         if end <= start + 1:
        #             continue
        #         items_to_merge = [self.item_seq[i] for i in range(start + 1, end)]
        #         self.item_seq[start].merge(items_to_merge, self.resourcepool)

    def splitAndMergeInRect(self, rect: QRectF):
        """Only work on LazyGraphicsPixmapItem

        Args:
            rect (QRectF): _description_
        """
        start_i = self.bisectY(rect.top())
        end_i = self.bisectY(rect.bottom())

        if start_i <= 0 or end_i <= 0:
            return

        # split the first & last item
        first_item = self.item_seq[start_i - 1]

        if not isinstance(first_item, LazyGraphicsPixmapItem):
            return

        need_split_last_item = end_i < len(self.item_seq)

        last_item_id = min(len(self.item_seq), end_i) - 1

        last_item: LazyGraphicsPixmapItem = self.item_seq[last_item_id]

        first_item: LazyGraphicsPixmapItem
        first_split_point = QPointF(0, rect.topLeft().y() - first_item.y())
        splited_first_item = first_item.vsplit(first_split_point, self.resourcepool)

        if need_split_last_item:
            last_split_point = QPointF(0, rect.bottomLeft().y() - last_item.y())
            last_item.vsplit(last_split_point, self.resourcepool)

        # merge all items into one item
        # print(
        #     f"merge from {start_i + 1} to {last_item_id + 1}, {len(self.item_seq)} in total"
        # )
        items_to_merge = [
            self.item_seq[i] for i in range(start_i + 1, last_item_id + 2)
        ]
        splited_first_item.merge(items_to_merge, self.resourcepool)

        # print("merge done")

        first_item.setSelected(False)
        splited_first_item.setSelected(True)

    def selectItemInRect(self, rect: QRectF):
        pass
        # start_i = self.bisectY(rect.top())
        # end_i = self.bisectY(rect.bottom())

        # if start_i <= 0 or end_i <= 0:
        #     return

        # for i in range(start_i - 1, min(len(self.item_seq), end_i)):
        #     item = self.item_seq[i]
        #     # item.setSelected(True)

    def deselectItemInRect(self, rect: QRectF):
        pass
        # start_i = self.bisectY(rect.top())
        # end_i = self.bisectY(rect.bottom())

        # if start_i <= 0 or end_i <= 0:
        #     return

        # for i in range(start_i - 1, min(len(self.item_seq), end_i)):
        #     item = self.item_seq[i]
        # item.setSelected(False)

    def bisect(self, item: QGraphicsItem) -> int:
        return bisect_left(
            SequentialGraphicsScene.ItemList(self.item_seq, key=lambda x: x.y()),
            item.y(),
        )

    def bisectY(self, y: float) -> int:
        return bisect_left(
            SequentialGraphicsScene.ItemList(self.item_seq, key=lambda x: x.y()),
            y,
        )

    def resetYAfter(self, index):
        if len(self.item_seq) > 0 and index < len(self.item_seq):
            y = self.item_seq[0].y() if index > 0 else 0
            for i in range(len(self.item_seq)):
                if i < index:
                    y += self.item_seq[i].size.height()
                else:
                    self.item_seq[i].setY(y)
                    y += self.item_seq[i].size.height()
        self.view_scene_connect.request_paint_signal.emit()

    def maskDisplayWithRange(self, displayRange: Range):
        for i in range(len(self.item_seq)):
            enabled = True
            if displayRange.isValid():
                enabled = i >= displayRange.start and i <= displayRange.end
            self.item_seq[i].setEnabled(enabled)
            self.item_seq[i].setVisible(enabled)

        first_id = (
            max(0, min(displayRange.start, len(self.item_seq) - 1))
            if displayRange.isValid()
            else 0
        )
        start_y = self.item_seq[first_id].y()

        last_id = (
            max(0, min(displayRange.end, len(self.item_seq) - 1))
            if displayRange.isValid()
            else len(self.item_seq) - 1
        )
        end_y = self.item_seq[last_id].y() + self.item_seq[last_id].size.height()
        return Range(start_y, end_y)


class LazySequentialCanvas(LazyCanvas):
    def __init__(self, resourcepool, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QHBoxLayout())
        self.scene: SequentialGraphicsScene = SequentialGraphicsScene(resourcepool)
        self.view: LazyGraphicsView = LazyGraphicsView(resourcepool, self.scene, self)
        self.layout().addWidget(self.view)
