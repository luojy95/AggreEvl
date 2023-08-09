from PySide6.QtWidgets import (
    QStyleOptionSlider,
    QSlider,
    QSizePolicy,
    QStyle,
    QWidget,
    QApplication,
)
from PySide6.QtGui import QPainter, QPaintEvent, QBrush, QPalette, QMouseEvent
from PySide6.QtCore import QRect, Qt, QSize, Signal


class RangeSlider(QWidget):
    # signals
    rangeChanged: Signal = Signal(object)

    def __init__(self, parent=None, orientation=Qt.Orientation.Horizontal):
        super().__init__(parent)

        self.first_position = 0
        self.second_position = 100

        self.opt = QStyleOptionSlider()
        self.opt.minimum = 0
        self.opt.maximum = 100

        self.orientation = orientation

        if self.orientation == Qt.Orientation.Horizontal:
            self.setSizePolicy(
                QSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider
                )
            )
        else:
            self.setSizePolicy(
                QSizePolicy(
                    QSizePolicy.Fixed, QSizePolicy.Expanding, QSizePolicy.Slider
                )
            )

    def setRangeLimit(self, minimum: int, maximum: int):
        self.opt.minimum = minimum
        self.opt.maximum = maximum

    def setRange(self, start: int, end: int):
        self.first_position = start
        self.second_position = end
        self.rangeChanged.emit(self.getRange())

    def getRange(self):
        return (self.first_position, self.second_position)

    def setTickPosition(self, position: QSlider.TickPosition):
        self.opt.tickPosition = position

    def setTickInterval(self, ti: int):
        self.opt.tickInterval = ti

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)

        # Draw rule
        self.opt.initFrom(self)
        self.opt.rect = self.rect()
        self.opt.orientation = self.orientation
        self.opt.sliderPosition = 0
        self.opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderTickmarks

        # Draw GROOVE
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        # Draw INTERVAL
        color = self.palette().color(QPalette.Highlight)
        color.setAlpha(160)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)

        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, self.opt, QStyle.SC_SliderGroove
        )

        if self.orientation == Qt.Orientation.Horizontal:
            self.opt.sliderPosition = self.first_position
            x_left_handle = (
                self.style()
                .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
                .right()
            )

            self.opt.sliderPosition = self.second_position
            x_right_handle = (
                self.style()
                .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
                .left()
            )

            selection = QRect(
                x_left_handle,
                groove_rect.y(),
                x_right_handle - x_left_handle,
                groove_rect.height(),
            ).adjusted(-1, 1, 1, -1)

        else:
            self.opt.sliderPosition = self.first_position
            y_top_handle = (
                self.style()
                .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
                .bottom()
            )

            self.opt.sliderPosition = self.second_position
            y_bottom_handle = (
                self.style()
                .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
                .top()
            )

            selection = QRect(
                groove_rect.x(),
                y_top_handle,
                groove_rect.width(),
                y_bottom_handle - y_top_handle,
            ).adjusted(1, -1, -1, 1)

        painter.drawRect(selection)

        # Draw first handle
        self.opt.subControls = QStyle.SC_SliderHandle
        self.opt.sliderPosition = self.first_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        # Draw second handle
        self.opt.sliderPosition = self.second_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

    def mousePressEvent(self, event: QMouseEvent):
        self.opt.sliderPosition = self.first_position
        self._first_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, event.pos(), self
        )

        self.opt.sliderPosition = self.second_position
        self._second_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, event.pos(), self
        )

    def mouseMoveEvent(self, event: QMouseEvent):
        distance = self.opt.maximum - self.opt.minimum

        current_pos = (
            event.pos().x()
            if self.orientation == Qt.Orientation.Horizontal
            else event.pos().y()
        )

        total_length = (
            self.rect().width()
            if self.orientation == Qt.Orientation.Horizontal
            else self.rect().height()
        )

        pos = self.style().sliderValueFromPosition(
            0, distance, current_pos, total_length
        )

        if self._first_sc == QStyle.SC_SliderHandle:
            if pos <= self.second_position:
                self.first_position = pos
                self.rangeChanged.emit(self.getRange())
                self.update()
                return

        if self._second_sc == QStyle.SC_SliderHandle:
            if pos >= self.first_position:
                self.second_position = pos
                self.rangeChanged.emit(self.getRange())
                self.update()

    def sizeHint(self):
        """override"""
        SliderLength = 84
        TickSpace = 5

        w = SliderLength
        h = self.style().pixelMetric(QStyle.PM_SliderThickness, self.opt, self)

        if (
            self.opt.tickPosition
            and QSlider.TicksAbove
            or self.opt.tickPosition
            and QSlider.TicksBelow
        ):
            h += TickSpace

        if self.orientation == Qt.Orientation.Vertical:
            t = w
            w = h
            h = t

        return self.style().sizeFromContents(
            QStyle.CT_Slider, self.opt, QSize(w, h), self
        )
