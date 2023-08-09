from gui.widgets.lazyCanvas.config.common import Crop


class LazyGraphicsPixmapItemConfig:
    def __init__(
        self,
        brightness=1,
        contrast=1,
        colortemp=-1,
        horicrop=Crop(),
        horiflip=False,
        vertflip=False,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.colortemp = colortemp
        self.horicrop = horicrop
        self.horiflip = horiflip
        self.vertflip = vertflip
