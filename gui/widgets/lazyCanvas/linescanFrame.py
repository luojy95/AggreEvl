from camera.linescan import LineScan
from gui.widgets.lazyCanvas.lazyGraphicsItem import LazyGraphicsPixmapItem


class LinescanFrame(LazyGraphicsPixmapItem):
    def __init__(
        self,
        camera: LineScan,
        frame_id,
        width,
        height,
        original_width,
        original_height,
        parent=None,
    ):
        super().__init__(width, height, original_width, original_height, parent)
        self.camera: LineScan = camera
        self.fid = frame_id
        self.path = camera.get_filename(self.fid)
        self.sourcePath = self.path
