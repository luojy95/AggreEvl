import sys

sys.path.append(".")

import numpy as np
import cv2
from scipy.interpolate import interp2d

import utils
from camera.linescan import LineScan
from tools.writer import writer
from mytest.test_utils import print_stat_info

line_dir = "data/line/2"

linescan = LineScan(line_dir)

data, info = linescan.get_multi(1780, 1810)
data = data[:, ::-1]

# utils.visualize(data, "processed_linescan.png")

# line_scan.visualize(img, f"{desc.id}_linescan_original.png")
hls = cv2.cvtColor(utils.to_uint8_img(data, require_normalize=False), cv2.COLOR_RGB2HLS)
lightness = hls[:, :, 1]
# writer.write_image(f"{desc.id}_lightness.png", lightness)
utils.visualize(lightness)
_, thresh = cv2.threshold(lightness, 80, 255, cv2.THRESH_BINARY)

h, w = thresh.shape
min_obj_size = (min(h, w) // 50) ** 2
min_hole_size = 100
mask_linescan = utils.get_clean_mask(thresh, min_obj_size, min_hole_size)

utils.visualize(mask_linescan)
