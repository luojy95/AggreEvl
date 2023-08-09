import sys

sys.path.append(".")

import numpy as np
import cv2
from scipy.interpolate import interp2d

import utils
from camera.gocator import GoCator
from camera.linescan import LineScan

from tools.writer import writer
from mytest.test_utils import print_stat_info

gocator_dir = "data/gocator/5"

go = GoCator(gocator_dir, reversed=False)

# dfs = go.get_multi(0, go.data_size())

# xs, ys, depth = go.get_scaled_data(dfs=dfs, use_interp=True, keep_inf=True)

line_dir = "data/line/5"

linescan = LineScan(line_dir)

# (aligned_depths, crop_start, crop_end,) = go.get_projected_depth_map_to_line_scan(
#     linescan, dfs, cropped=False, use_interp=True
# )

# aligned_depths = go.fix_depth(aligned_depths, denoise=True)

# mask = aligned_depths == np.inf
# valid = np.logical_not(mask)

# aligned_image = GoCator.get_image_from_depths(aligned_depths)

# utils.visualize(aligned_image, f"aligned_depth.png")
# h, w = aligned_image.shape
# terrain_img = cv2.resize(cv2.resize(aligned_image, (w // 100, h // 100)), (w, h))
# terrain = np.zeros_like(aligned_depths, dtype=np.float32)

# interp_data = (
#     terrain_img[valid].astype(np.float32)
#     / 255.0
#     * (aligned_depths[valid].max() - aligned_depths[valid].min())
#     + aligned_depths[valid].min()
# )
# terrain[mask] = np.inf
# terrain[valid] = aligned_depths[valid] - interp_data

# filterout_mask = np.logical_or(
#     terrain <= np.percentile(terrain[valid], 2),
#     terrain >= np.percentile(terrain[valid], 98),
# )

# terrain[filterout_mask] = np.inf

# terrain_img = GoCator.get_image_from_depths(terrain)

# utils.visualize(terrain_img, f"deterrain_gocator.png")

# print_stat_info(terrain_img.astype(np.float32).ravel())

# _, thresh = cv2.threshold(
#     terrain_img,
#     np.percentile(terrain_img.astype(np.float32).ravel(), 75),
#     255,
#     cv2.THRESH_BINARY,
# )

# h, w = thresh.shape
# min_obj_size = (min(h, w) // 100) ** 2
# min_hole_size = (min(h, w) // 100) ** 2
# mask_gocator = utils.get_clean_mask(thresh, min_obj_size, min_hole_size)

# print(crop_start, crop_end)
# utils.visualize(mask_gocator, f"processed_gocator.png")

linedata, info = linescan.get_multi(780, 1500)
linedata = linedata[:, ::-1]

utils.visualize(linedata, "linescane_orig.png")

crop_start = 979
crop_end = 3093
# line_scan.visualize(img, f"{desc.id}_linescan_original.png")
lab = cv2.cvtColor(
    utils.to_uint8_img(linedata, require_normalize=False), cv2.COLOR_RGB2LAB
)
l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl, a, b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
utils.visualize(l_channel, "lightness.png")


utils.visualize(cl, "enhanced_lightness.png")

h, w = l_channel.shape
avg_lighting = cv2.resize(cv2.resize(cl, (w // 100, h // 100)), (w, h))
da_l = utils.to_uint8_img(1.0 - utils.normalize(cl - avg_lighting))
utils.visualize(da_l, "dal.png")

_, thresh = cv2.threshold(cl, 120, 255, cv2.THRESH_BINARY)
h, w = thresh.shape
min_obj_size = (min(h, w) // 50) ** 2
min_hole_size = 50
mask_linescan = utils.get_clean_mask(thresh, min_obj_size, min_hole_size)
mask_linescan[:, :crop_start] = 0
mask_linescan[:, crop_end:] = 0
utils.visualize(mask_linescan, "mask_linescan_0.png")

_, thresh = cv2.threshold(da_l, 120, 255, cv2.THRESH_BINARY)
h, w = thresh.shape
min_obj_size = (min(h, w) // 50) ** 2
min_hole_size = (min(h, w) // 50) ** 2
mask_linescan = utils.get_clean_mask(thresh, min_obj_size, min_hole_size)

mask_linescan[:, :crop_start] = 0
mask_linescan[:, crop_end:] = 0
utils.visualize(mask_linescan, "mask_linescan.png")

# combine = np.concatenate([mask_gocator, mask_linescan], axis=0)
# utils.visualize(combine, "processed_combine.png")
#  ==================================================================
# utils.visualize(filled_mask)

# not_inf = depth != np.inf
# is_inf = np.logical_not(not_inf)
# max_v = 1e5

# depth[is_inf] = max_v
# print_stat_info(depth)


# gradx, grady = np.gradient(depth, edge_order=2)
# print_stat_info(gradx)
# print_stat_info(grady)

# grad = np.abs(gradx) + np.abs(grady)

# print_stat_info(grad)

# grad[grad > 7] = 0.0


# img = utils.to_uint8_img(grad, require_normalize=True)

# _, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)


# writer.write_image("grad.png", thresh)

# valid = depth[msk]

# print(valid.max(), valid.min(), valid.mean(), valid.std())
