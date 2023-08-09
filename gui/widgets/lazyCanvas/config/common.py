from PySide6 import QtCore


class Range:
    def __init__(self, start=0, end=0) -> None:
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"Range from {self.start} to {self.end}"

    def isValid(self):
        return self.start < self.end

    def size(self):
        return self.end - self.start


class Scale:
    def __init__(self, sx=1.0, sy=1.0) -> None:
        self.x = sx
        self.y = sy

    def __str__(self) -> str:
        return f"Scale with x - {self.x}, y - {self.y}"

    def toQPointF(self) -> QtCore.QPointF:
        return QtCore.QPointF(self.x, self.y)


class Crop:
    def __init__(self, top=0, bottom=0, left=0, right=0) -> None:
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"Crop with top - {self.top}, bottom - {self.bottom}, left - {self.left}, right - {self.right}"
