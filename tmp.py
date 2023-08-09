import os
import cv2
import numpy as np
import tqdm
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager

from collections import namedtuple

import utils
from camera.gocator import GoCator
from camera.linescan import LineScan
from common import common, FrameRange
from tools.writer import writer
from tools.loader import loader

scan_name = "5_cut4"
gocator_root = "gocator"
linescan_root = "line"

gocator_dir = os.path.join(common.data_path, gocator_root, scan_name)
linescan_dir = os.path.join(common.data_path, linescan_root, scan_name)

LGATaskDesc = namedtuple(
    "LGATaskDesc", ["line_scan_range", "gocator_range", "is_continuous"]
)


class QueryServer:
    def __init__(self, template, sample):
        self.template = template.copy()
        self.sample = sample.copy()
        self.result = {}

    def get(self):
        return self.template, self.sample, self.result

    def set(self, value):
        size, result = value
        self.result[size] = result


def query_if_size_match(query_server, size, start_col):
    template, sample, _ = query_server.get()
    tw, th = size
    resized_sample = cv2.resize(sample, size).astype(np.float32)
    ph, pw = template.shape
    resized_sample = resized_sample[
        :, tw // 2 - min(tw, pw) // 2 : tw // 2 + min(tw, pw) // 2
    ]
    startw = pw // 2 - min(pw, tw) // 2
    endw = pw // 2 + min(pw, tw) // 2

    max_iou = 0.0
    opt = None

    # coarse to fine search
    granularity_lev = 8
    end_col = ph

    while granularity_lev > 1:
        start_col_opt = start_col
        end_col_opt = end_col
        for starth in range(start_col, end_col, granularity_lev):
            endh = min(starth + th, ph)
            outh = endh - starth
            template_float = template[starth:endh, startw:endw].astype(np.float32)

            iou = utils.calc_iou(template_float, resized_sample[:outh])
            if iou > max_iou:
                max_iou = iou
                opt = (iou, starth, startw, outh, endw - startw)
                start_col_opt = starth - granularity_lev
                end_col_opt = starth + granularity_lev

        start_col = start_col_opt
        end_col = end_col_opt
        granularity_lev = granularity_lev // 2

    query_server.set((size, opt))


def align(desc: LGATaskDesc):
    gocator = GoCator(data_dir=gocator_dir)
    line_scan = LineScan(data_dir=linescan_dir)

    """ GoCator """
    dfs = gocator.get_multi(desc.gocator_range.start, desc.gocator_range.end)
    gocator.visualize(dfs, "gocator_original.png")

    aligned_depths, startw, endw = gocator.get_projected_depth_map_to_line_scan(
        line_scan, dfs, cropped=False
    )
    aligned_image = GoCator.get_image_from_depths(aligned_depths)
    line_scan.visualize(aligned_image, "gocator_project.png")

    aligned_mask = aligned_image > 0

    h, w = aligned_mask.shape
    min_obj_size = (min(h, w) // 100) ** 2
    min_hole_size = 100
    mask_gocator = utils.get_clean_mask(aligned_mask, min_obj_size, min_hole_size)

    writer.write_image("mask_gocator_test.png", utils.to_uint8_img(mask_gocator))

    """ Line Scan """
    start_frame = desc.line_scan_range.start
    end_frame = desc.line_scan_range.end
    img, _ = line_scan.get_multi(start_frame, end_frame)
    line_scan.visualize(img, "linescan_original.png")

    hls = cv2.cvtColor(
        utils.to_uint8_img(img, require_normalize=False), cv2.COLOR_RGB2HLS
    )
    lightness = hls[:, :, 1]
    writer.write_image("lightness.png", lightness)

    # pyramid = utils.pyramid(lightness, level=0)

    _, thresh = cv2.threshold(lightness, 80, 255, cv2.THRESH_BINARY)

    h, w = thresh.shape
    min_obj_size = (min(h, w) // 100) ** 2
    min_hole_size = 100
    mask_linescan = utils.get_clean_mask(thresh, min_obj_size, min_hole_size)

    writer.write_image("lightness_threshed.png", mask_linescan)

    # crop to match goccator mask
    h, w = mask_linescan.shape
    mask_linescan = line_scan.visualize(mask_linescan, export_path="mask_linescan.png")

    gh, outw = mask_gocator.shape
    lh, outw = mask_linescan.shape

    mask_linescan[:, :startw] = 0
    mask_linescan[:, endw - 1 :] = 0

    combined = np.zeros([gh, outw, 3], dtype=np.uint8)
    template = mask_gocator
    combined[:, :, 2] = utils.to_uint8_img(template)

    ph, pw = template.shape

    current_row = 0
    rows_per_frame = 50
    total_rows = (end_frame - start_frame) * rows_per_frame
    start_template_w = 0
    start_template_h = 0
    fixed_width = -1

    frames = [combined.copy()]

    vmap = np.arange(gh).astype(np.float32)
    vmap_not_updated = np.ones(gh, dtype=np.bool8)

    while current_row < total_rows:
        row_interval = 500 if current_row == 0 else 200
        sample = mask_linescan[current_row : current_row + row_interval]

        sh, sw = sample.shape

        opt = None
        if not desc.is_continuous or current_row == 0:
            scalesx = np.array([1.0])  # scalesx = np.linspace(0.9, .95, 5)
            scalesy = np.linspace(1, 3, 20)
            target_ws = np.round(sw * scalesx).astype(np.int32).tolist()
            target_hs = np.round(sh * scalesy).astype(np.int32).tolist()

            BaseManager.register("QueryServer", QueryServer)
            manager = BaseManager()
            manager.start()
            server = manager.QueryServer(template, sample)
            pool = Pool()

            for target_w in target_ws:
                for target_h in target_hs:
                    pool.apply_async(
                        query_if_size_match,
                        args=(server, (target_w, target_h), start_template_h),
                    )
            pool.close()
            pool.join()

            _, _, r = server.get()

            ranks = sorted(list(r.keys()), key=lambda x: r[x][0], reverse=True)

            size = ranks[0]
            opt = r[size]
            iou, starth, startw, h, w = r[size]

            # v' -> current_row + (v' - starth) * row_interval / h
            vmap[starth : starth + h] = (
                (vmap[starth : starth + h] - starth)
            ) * row_interval / h + current_row

            vmap_not_updated[starth : starth + h] = False

            scaled_sample = cv2.resize(sample, size).astype(np.float32)[:h]

            combined[starth : starth + h, startw : startw + w, 1] = utils.to_uint8_img(
                scaled_sample
            )
            start_template_w = startw
            start_template_h = starth + h
            fixed_width = w
        else:  # a much faster version for continuous data captured
            scalesy = np.linspace(0.8, 3.2, 24)
            target_hs = np.round(sh * scalesy).astype(np.int32).tolist()

            max_iou = 0.0
            for target_h in target_hs:
                resized_sample = cv2.resize(sample, (fixed_width, target_h)).astype(
                    np.float32
                )
                th, tw = resized_sample.shape
                startw = start_template_w
                endw = start_template_w + fixed_width

                starth = start_template_h
                endh = min(starth + th, ph)
                outh = endh - starth
                template_float = template[starth:endh, startw:endw].astype(np.float32)

                iou = utils.calc_iou(template_float, resized_sample[:outh])
                if iou > max_iou:
                    max_iou = iou
                    opt = (
                        iou,
                        starth,
                        startw,
                        endh - starth,
                        endw - startw,
                        resized_sample[:outh],
                    )
            if opt is None:
                writer.write_image(f"err_results_{current_row}.png", combined)
                writer.write_image("err_sample.png", sample)
                return
            iou, starth, startw, h, w, scaled_sample = opt

            vmap[starth : starth + h] = (
                (vmap[starth : starth + h] - start_template_h)
            ) * row_interval / h + current_row
            vmap_not_updated[starth : starth + h] = False

            combined[starth : starth + h, startw : startw + w, 1] = utils.to_uint8_img(
                scaled_sample
            )
            start_template_h = starth + h

        current_row += row_interval

        if (current_row % 100) == 0:
            # writer.write_image(f"results_{current_row}.png", combined)
            frames.append(combined.copy())
            print(f"{current_row}/{total_rows} done.")

    writer.write_image(f"results.png", combined)
    writer.write_images_to_gif(f"animation.gif", frames)

    vmap[vmap_not_updated] = -1
    print(vmap.max(), total_rows)

    h, w = aligned_depths.shape
    vy, vx = np.where(aligned_depths != np.inf)
    us = np.arange(w)[vx][:, np.newaxis]
    vs = np.arange(h)[vy][:, np.newaxis]
    uv = np.concatenate([us, vs], axis=1)

    xyz = gocator.get_reprojected_points_from_line_scan(line_scan, dfs, aligned_depths)
    num_points = xyz.shape[0]

    querybook = np.zeros([num_points, 2], dtype=np.int32)
    querybook[:, 0] = uv[:, 0]
    querybook[:, 1] = vmap[uv[:, 1]]

    print(querybook.max(axis=0), querybook.min(axis=0))

    writer.write_point_cloud("outputs/reprojected.pcd", xyz)
    np.save("outputs/querybook.npy", querybook)


def vis_align_benchmark(querybook, pcd, labels):
    points = np.asarray(pcd.points)
    colors = np.zeros_like(points) + 0.7
    # print(querybook.shape, points.shape, labels.shape)
    querybook_ids = querybook.astype(np.int32)
    valid_labels = labels[querybook_ids[:, 1], querybook_ids[:, 0]].astype(np.uint32)
    valid_id_mask = valid_labels > 0
    valid_labels = valid_labels[valid_id_mask]

    num_included_particles = np.unique(valid_labels).shape[0]
    print(num_included_particles, np.max(valid_labels))
    checkings = np.zeros(np.max(valid_labels) + 1)
    for i, id in enumerate(np.unique(valid_labels)):
        checkings[id] = i + 1

    valid_labels = checkings[valid_labels]
    print(valid_labels)

    colors[valid_id_mask] = utils.get_rgba(valid_labels / (num_included_particles + 1))[
        :, :-1
    ]

    # colors = (colors * 255).astype(np.uint8)

    # print(colors)

    pcd = writer.write_point_cloud("segmented.ply", points, colors)
    utils.visualize_pcds([pcd])


if __name__ == "__main__":
    task_desc_1 = LGATaskDesc(
        line_scan_range=FrameRange(1028, 1086),
        gocator_range=FrameRange(1, 3),
        is_continuous=True,
    )

    task_desc_2 = LGATaskDesc(
        line_scan_range=FrameRange(1440, 1480),
        gocator_range=FrameRange(13, 15),
        is_continuous=False,
    )
    align(task_desc_1)

    # querybook_path = os.path.join(common.output_path, "querybook.npy")
    # pcd_path = os.path.join(common.output_path, "reprojected.pcd")
    # labels_path = os.path.join(common.output_path, "aligned_labels.npy")

    # vis_align_benchmark(
    #     loader.load_data(querybook_path),
    #     loader.load_data(pcd_path),
    #     loader.load_data(labels_path),
    # )
