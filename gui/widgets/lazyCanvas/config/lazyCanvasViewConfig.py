from gui.widgets.lazyCanvas.config.common import Range, Scale, Crop


class LazyCanvasViewConfig:
    def __init__(
        self,
        displayRange: Range = Range(),
        initialScale: Scale = Scale(),
    ) -> None:
        self.displayRange = displayRange
        self.initialScale = initialScale
