import os
import numpy as np
import tqdm
import cv2

import utils
from camera.linescan import LineScan
from common import common, FrameRange
from tools.writer import writer
from tools.loader import loader


def vis_all_labels(name, label_data):
    vis = np.zeros((h, w), dtype=np.uint8)

    vis += np.sum(label_data[0], axis=0) > 0
    vis *= 255

    return writer.write_image(name, vis)


scan_name = "5_cut4"
linescan_root = "line"
linescan_dir = os.path.join(common.data_path, linescan_root, scan_name)
line_scan = LineScan(linescan_dir)


label_root = "2ds"
label_name = "segment_01.npy"
label_image_name = "segment_01.png"
label_path = os.path.join(common.data_path, label_root, label_name)
label_img_path = os.path.join(common.data_path, label_root, label_image_name)


label_image = loader.load_image(label_img_path)[:, ::-1]
lh, lw = label_image.shape[:2]


camera_region = FrameRange(1028, 1086)
camera_image, _ = line_scan.get_multi(camera_region.start, camera_region.end)
ch, cw = camera_image.shape[:2]

label_hls_mask = utils.get_hls_mask(label_image)
camera_hls_mask = utils.get_hls_mask(camera_image)

target_h = 3000
sh = -70
sw = 205
outh = min(target_h, ch) - np.abs(sh)
config = (target_h, sw, sh, outh)

label_hls_mask = cv2.resize(label_hls_mask, (lw, target_h))
label_image_r = cv2.resize(label_image, (lw, target_h))
blend = np.zeros([ch, cw, 3], dtype=np.uint8)
# blend[:, :, 1] = cv2.cvtColor(camera_image * 255, cv2.COLOR_RGB2GRAY)
blend[max(0, sh) : sh + outh, sw : sw + lw, 1] = label_image_r[-sh:outh, :, 1]
blend[max(0, sh) : sh + outh, sw : sw + lw, 2] = label_hls_mask[-sh:outh]

out_mask = np.zeros([ch, cw], dtype=np.int32)

dat = np.load(label_path)

_, num_ballast, h, w = dat.shape

for i in range(num_ballast):
    ballast_mask = dat[0, i][:, ::-1]
    resized_mask = cv2.resize(ballast_mask.astype(np.uint8) * 255, (lw, target_h))
    blend[max(0, sh) : sh + outh, sw : sw + lw, 0] |= resized_mask[-sh:outh]
    mask = np.zeros([ch, cw], dtype=np.uint8)
    mask[max(0, sh) : sh + outh, sw : sw + lw] |= resized_mask[-sh:outh]
    out_mask[mask > 100] = i + 1

writer.write_image("blend.png", blend)

np.save("outputs/aligned_labels.npy", out_mask)
