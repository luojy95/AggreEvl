from camera.linescan import LineScan
from camera.gocator import GoCator
from gui.widgets.lazyCanvas.lazyCanvas import LazyCanvas
from gui.widgets.lazyCanvas.lazyGraphicsItem import LazyGraphicsPixmapItem
from gui.widgets.lazyCanvas.linescanFrame import LinescanFrame
from gui.widgets.lazyCanvas.gocatorFrame import GocatorFrame, ProjectedGocatorFrame

default_scan = 5


def test_linescan(
    canvas: LazyCanvas,
    target_path: str = f"data/line/{default_scan}",
    target_width=1024,
):
    camera = LineScan(target_path)

    preload_data, _ = camera.get(0)

    h, w = preload_data.shape[:-1]

    target_w = target_width
    target_h = int(target_w * 1.0 * h / w)

    for i in range(camera.data_size()):
        item = LinescanFrame(camera, i, target_w, target_h, w, h)
        item.setPos(0, target_h * i)
        item.release()
        canvas.addItem(item)


def test_gocator(
    canvas: LazyCanvas, target_path: str = f"data/gocator/xlsx/{default_scan}"
):
    camera = GoCator(target_path)

    dfs = camera.get(0)
    target_w = 1024
    _, _, data = camera.get_scaled_data(
        dfs, force_width=target_w, use_interp=True, keep_inf=True
    )
    preload_data = camera.get_image_from_depths(data)

    h, w = preload_data.shape

    target_h = target_w * 1.0 * h / w

    global_approx_min = None
    global_approx_max = None

    for i in range(camera.data_size()):
        item = GocatorFrame(camera, i, target_w, target_h, w, h)
        if global_approx_min is None or global_approx_max is None:
            global_approx_min, global_approx_max = item.estimate_approx_value_range()
        item.approx_min_value, item.approx_max_value = (
            global_approx_min,
            global_approx_max,
        )
        item.approx_min_value_limit, item.approx_max_value_limit = (
            global_approx_min,
            global_approx_max,
        )
        item.setPos(0, target_h * i)
        item.release()
        canvas.addItem(item)


def test_projected_gocator(
    canvas: LazyCanvas,
    gocator_path=f"data/gocator/xlsx/{default_scan}",
    linescan_path=f"data/line/{default_scan}",
):
    gocator = GoCator(gocator_path)
    linescan = LineScan(linescan_path)

    dfs = gocator.get(0)
    data_h, data_w = dfs["data"].to_numpy()[:, 1:].shape
    x_resolution = dfs["info"]["XResolution"].values[0]
    y_resolution = dfs["info"]["YResolution"].values[0]
    aspect_ratio = (y_resolution * data_h) / (x_resolution * data_w)
    h_calculated = round(data_w * aspect_ratio)
    h, w = h_calculated, linescan.res

    target_w = 1024
    target_h = round(target_w * 1.0 * h / w)

    global_approx_min = -2000
    global_approx_max = 0

    for i in range(gocator.data_size()):
        item = ProjectedGocatorFrame(gocator, linescan, i, target_w, target_h, w, h)
        if global_approx_min is None or global_approx_max is None:
            global_approx_min, global_approx_max = item.estimate_approx_value_range()
        item.approx_min_value, item.approx_max_value = (
            global_approx_min,
            global_approx_max,
        )
        item.approx_min_value_limit, item.approx_max_value_limit = (
            global_approx_min,
            global_approx_max,
        )
        item.setPos(0, target_h * i)
        item.release()
        canvas.addItem(item)
    output_min = gocator.get_sensor_value_from_distance_to_camera(
        dfs, -global_approx_min
    )
    output_max = gocator.get_sensor_value_from_distance_to_camera(
        dfs, -global_approx_max
    )
    return output_min, output_max


def test(canvas: LazyCanvas):
    test_gocator(canvas)
