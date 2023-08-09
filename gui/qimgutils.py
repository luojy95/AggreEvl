from PySide6 import QtGui
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QSize, QRect, Qt
from typing import Tuple, List

import numpy as np
from PIL import Image


class qImgUtils:
    """QImage utilities to operate qimages"""

    @staticmethod
    def load_qimg(path):
        qimg = QtGui.QImage()
        with open(path, "rb") as f:
            qimg.loadFromData(f.read())
        return qimg

    @classmethod
    def vsplit(cls, qimg: QImage, ratio: float) -> Tuple[QImage, QImage]:
        """vertically split the qimage with ratio

        Args:
            qimg (QImage): _description_
            ratio (float): _description_

        Returns:
            tuple[QImage, QImage]: _description_
        """
        size = qimg.size()

        height_up = int(size.height() * 1.0 * ratio)

        qimage_1 = qimg.copy(QRect(0, 0, size.width(), height_up))

        qimage_2 = qimg.copy(
            QRect(0, height_up, size.width(), size.height() - height_up)
        )

        return qimage_1, qimage_2

    @classmethod
    def vstack(cls, qimgs: List[QImage]) -> QImage:
        """vertically stack a list of qimages

        Args:
            qimgs (list[QImage]): _description_

        Raises:
            ValueError: _description_

        Returns:
            QImage: _description_
        """
        if len(qimgs) == 0:
            return None

        if len(qimgs) == 1:
            return qimgs[0]

        size = qimgs[0].size()
        height = size.height()
        for qimg in qimgs[1:]:
            if size.width() != qimg.size().width():
                raise ValueError("input image for vstack should have the same width")
            height += qimg.size().height()

        output = QImage(size.width(), height, QImage.Format_RGB888)

        painter = QPainter()
        painter.begin(output)

        y = 0
        for qimg in qimgs:
            painter.drawImage(0, y, qimg)
            y += qimg.size().height()

        painter.end()

        return output

    @classmethod
    def resize(cls, qimage: QImage, size: QSize):
        return qimage.scaled(size, Qt.IgnoreAspectRatio)

    @classmethod
    def from_np(cls, img_np: np.ndarray, format: QtGui.QImage.Format) -> QtGui.QImage:
        return QtGui.QImage(
            img_np, img_np.shape[1], img_np.shape[0], img_np.strides[0], format
        )

    @classmethod
    def to_np(cls, qimage: QImage) -> np.ndarray:
        W = qimage.width()
        H = qimage.height()
        C = qimage.depth()

        bits = qimage.bits()

        size = (H, W) if C == 1 else (H, W, C)

        bits.setsize(H * W * C)

        return np.frombuffer(bits, np.uint8).reshape(size)

    @classmethod
    def to_Image(cls, qimage: QImage) -> Image:
        return Image.fromqimage(qimage)
