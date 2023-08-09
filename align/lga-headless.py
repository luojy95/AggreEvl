import sys

sys.path.append(".")
import os
import cv2

import numpy as np
from PIL import Image
import json
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager

import matplotlib.cm as cm

from collections import namedtuple

import utils
from camera.gocator import GoCator
from camera.linescan import LineScan
from common import common, FrameRange
from tools.writer import writer
from tools.loader import loader
from tools.logger import Logger

import argparse

parser = argparse.ArgumentParser()

args = parser.parse_args()

logger = Logger(os.path.join(common.output_path, "log.txt"))

LGATaskParams = {
    "id": "0",
    "gocator_range": FrameRange(1, 3),
    "linescan_range": FrameRange(28, 86),
    "linescan_row_offset": 0,
    "is_first_task": True,
    "frame_interval_thorough_check": 20,
    "frame_interval_continuous_check": 8,
    "fixed_width": -1,
    "iou_thresh": 0.58,
}

LGATaskDescriptor = namedtuple(
    "LGATaskDescriptor", list(LGATaskParams.keys()), defaults=LGATaskParams.values()
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
    sh, _ = sample.shape
    tw, th = size
    scale = sh / th
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

            iou = utils.calc_iou(template_float, resized_sample[:outh]) * outh * scale
            if iou > max_iou:
                max_iou = iou
                opt = (iou, starth, startw, outh, endw - startw)
                start_col_opt = starth - granularity_lev
                end_col_opt = starth + granularity_lev

        start_col = start_col_opt
        end_col = end_col_opt
        granularity_lev = granularity_lev // 2

    query_server.set((size, opt))


gocator_mask_full = cv2.imread(
    "outputs-test/processed_gocator.png", cv2.IMREAD_GRAYSCALE
)
linescan_mask_full = cv2.imread(
    "outputs-test/mask_linescan_0.png", cv2.IMREAD_GRAYSCALE
)
num_gocator_frames = 17
search_range = gocator_mask_full.shape[0] // num_gocator_frames + 1
rows_per_linescan_frame = 50
num_linescan_frames = linescan_mask_full.shape[0] // rows_per_linescan_frame

print(gocator_mask_full.shape)


def LGATask(desc: LGATaskDescriptor):
    global gocator_mask_full, linescan_mask_full, search_range, num_linescan_frames
    logger.info(f"Start LGATask {desc.id}")

    gi = desc.gocator_range.start

    mask_gocator = gocator_mask_full[
        gi
        * search_range : min(
            desc.gocator_range.end * search_range, gocator_mask_full.shape[0]
        )
    ]

    # print(
    #     mask_gocator.shape,
    #     mask_gocator.min(),
    #     mask_gocator.max(),
    #     search_range,
    #     gocator_mask_full.shape,
    # )

    """ Line Scan """
    start_frame = desc.linescan_range.start

    gh, outw = mask_gocator.shape
    combined = np.zeros([gh, outw, 3], dtype=np.uint8)
    template = mask_gocator
    combined[:, :, 2] = utils.to_uint8_img(template)

    ph, pw = template.shape

    rows_per_frame = common.num_rows_per_linescan_frame

    start_template_w = 0
    start_template_h = 0
    fixed_width = desc.fixed_width

    logger.info("Start alignment loops.")

    covered_rows = 0

    results = {}

    scales = []
    while start_template_h < gh and start_frame < num_linescan_frames:
        row_interval = -1
        opt = None
        need_thorough_check = True
        if start_template_h != 0 or not desc.is_first_task:
            row_interval = rows_per_frame * desc.frame_interval_continuous_check

            sample_start = start_frame * rows_per_frame
            sample_end = min(
                num_linescan_frames * rows_per_frame, sample_start + row_interval
            )
            row_interval = min(row_interval, sample_end - sample_start)

            num_frames = row_interval // rows_per_frame

            sample = linescan_mask_full[sample_start:sample_end]

            sh, sw = sample.shape
            row_interval = sh

            scalesy = np.linspace(0.8, 1.6, 24)
            target_hs = np.round(sh * scalesy).astype(np.int32).tolist()

            max_iou = 0.0
            for th in target_hs:
                resized_sample = cv2.resize(sample, (fixed_width, th)).astype(
                    np.float32
                )
                startw = start_template_w
                endw = start_template_w + fixed_width

                starth = start_template_h
                endh = min(starth + th, ph)
                outh = endh - starth
                template_float = template[starth:endh, startw:endw].astype(np.float32)

                iou = (
                    utils.calc_iou(template_float, resized_sample[:outh])
                    * outh
                    * sh
                    / th
                )
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
                    covered_rows = np.floor(outh / th * row_interval).astype(np.int32)

            iou, starth, startw, h, w, scaled_sample = opt

            if iou >= desc.iou_thresh:
                logger.info(
                    f"Continuous check IOU ({start_frame:04d}-{start_frame + num_frames:04d}): {iou}"
                )
                print(scaled_sample.shape[0] / covered_rows)
                combined[
                    starth : starth + h, startw : startw + w, 1
                ] = utils.to_uint8_img(scaled_sample)
                if start_frame == desc.linescan_range.start:
                    results[
                        f"{start_frame}, {num_frames}, {desc.linescan_row_offset}"
                    ] = [starth, h, int(covered_rows)]
                else:
                    results[f"{start_frame}, {num_frames}, 0"] = [
                        starth,
                        h,
                        int(covered_rows),
                    ]

                start_frame += num_frames
                start_template_h = starth + h
                need_thorough_check = False

        if need_thorough_check:
            row_interval = rows_per_frame * desc.frame_interval_thorough_check

            sample_start = start_frame * rows_per_frame
            sample_end = min(
                num_linescan_frames * rows_per_frame, sample_start + row_interval
            )
            row_interval = min(row_interval, sample_end - sample_start)

            num_frames = row_interval // rows_per_frame

            sample = linescan_mask_full[sample_start:sample_end]

            sh, sw = sample.shape
            row_interval = sh

            scalesx = np.array([1.0])  # scalesx = np.linspace(0.9, 1.1, 5)
            scalesy = np.linspace(0.8, 1.6, 24)
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

            logger.info(
                f"Thorough check IOU ({start_frame:04d}-{start_frame+num_frames:04d}): {iou}"
            )

            covered_rows = np.floor(h / size[1] * row_interval).astype(np.int32)

            if start_frame == desc.linescan_range.start:
                results[f"{start_frame}, {num_frames}, {desc.linescan_row_offset}"] = [
                    starth,
                    h,
                    int(covered_rows),
                ]
            else:
                results[f"{start_frame}, {num_frames}, 0"] = [
                    starth,
                    h,
                    int(covered_rows),
                ]

            start_frame += num_frames

            scaled_sample = cv2.resize(sample, size).astype(np.float32)[:h]
            print(scaled_sample.shape[0] / covered_rows)

            combined[starth : starth + h, startw : startw + w, 1] = utils.to_uint8_img(
                scaled_sample
            )
            start_template_w = startw
            start_template_h = starth + h
            fixed_width = w

    logger.info("Alignment loop completed.")

    covered_frames = covered_rows // rows_per_frame
    covered_rows %= rows_per_frame
    next_start_frame = start_frame - num_frames + covered_frames

    writer.write_image(f"{desc.id}_result_overlapping.png", combined)

    logger.ok(f"LGATesk {desc.id} done.")

    return (
        next_start_frame,
        covered_rows,
        results,
        fixed_width,
        next_start_frame >= num_linescan_frames,
    )


if __name__ == "__main__":
    ckpt = {
        "task_id": 0,
        "start_gocator": 0,
        "start_linescan": 0,
        "frame_offset": 0,
        "fixed_width": -1,
    }

    ckpt_path = os.path.join(common.output_path, "ckpt.json")
    # if os.path.exists(ckpt_path):
    #     f = open(ckpt_path, "r")
    #     ckpt.update(json.load(f))
    #     f.close()

    results = {}
    default_result_path = os.path.join(common.output_path, "result_lga.json")
    if os.path.exists(default_result_path):
        f = open(default_result_path, "r")
        results.update(json.load(f))
        f.close()

    try:
        while True:
            tid = f"task-{ckpt['task_id']:03d}"
            task_desc = LGATaskDescriptor(
                id=tid,
                linescan_range=FrameRange(ckpt["start_linescan"], -1),
                linescan_row_offset=ckpt["frame_offset"],
                gocator_range=FrameRange(
                    ckpt["start_gocator"], ckpt["start_gocator"] + 2
                ),
                is_first_task=(ckpt["task_id"] == 0),
                fixed_width=ckpt["fixed_width"],
            )

            logger.info(str(ckpt))

            (
                start_linescan,
                frame_offset,
                results[tid],
                fixed_width,
                over,
            ) = LGATask(task_desc)

            ckpt["task_id"] += 1

            if not over:
                ckpt["start_gocator"] = int(ckpt["start_gocator"] + 2)
                ckpt["start_linescan"] = int(start_linescan)
                ckpt["frame_offset"] = int(frame_offset)
                ckpt["fixed_width"] = int(fixed_width)

            if over:
                break

    except RuntimeError as e:
        logger.error(str(e))

    finally:
        json.dump(ckpt, open(ckpt_path, "w+", encoding="utf-8"), indent=4)
        json.dump(results, open(default_result_path, "w+", encoding="utf-8"), indent=4)

    # desc = LGATaskDescriptor()
    # gocator = GoCator(data_dir=desc.gocator_dir)
    # line_scan = LineScan(data_dir=desc.linescan_dir)
    # sorted_keys = sorted(list(results.keys()))
    # depth_map = {}
    # minv = np.inf
    # maxv = -np.inf
    # for key in sorted_keys:
    #     depth_path = os.path.join(common.output_path, f"{key}_gocator_reprojected.txt")
    #     if not os.path.exists(depth_path):
    #         break
    #     depth = np.loadtxt(depth_path)
    #     depth_map[key] = depth
    #     mask = depth != np.inf
    #     minv = min(minv, np.percentile(depth[mask], 5))
    #     maxv = max(maxv, np.percentile(depth[mask], 95))
    # print(minv, maxv)
    # normalize = lambda d: (d - minv) / (maxv - minv)

    # def clr_depth(depth):
    #     clr = cm.gist_ncar(normalize(depth))
    #     return utils.to_uint8_img(clr, False)

    # imgId = 0
    # startRow = -1
    # cachedRow = 0
    # imgSeq = []
    # depthSeq = []
    # for key in sorted_keys:
    #     data = results[key]
    #     if data is None:
    #         break

    #     depth = depth_map[key]

    #     sorted_frame_keys = sorted(
    #         list(data.keys()),
    #         key=lambda x: [int(x.split(",")[0]), int(x.split(",")[-1])],
    #     )
    #     for frame_key in sorted_frame_keys:
    #         start_frame, length, offset = [int(s) for s in frame_key.split(",")]
    #         img, _ = line_scan.get_multi(start_frame, start_frame + length)
    #         oh, ow = img.shape[:2]
    #         starth, h, covered = data[frame_key]

    #         depth_frame = depth[starth : starth + h]

    #         img = img[offset : min(oh, offset + covered)]
    #         resized = cv2.resize(img, (ow, h))

    #         starth += cachedRow

    #         print(key, frame_key, starth, startRow, cachedRow)

    #         if startRow == -1:
    #             startRow = starth

    #         if starth - startRow > 5:
    #             output = np.concatenate(imgSeq, axis=0).astype(np.uint8)
    #             writer.write_image(f"sequence_{imgId:04d}.png", output)
    #             depth_out = np.concatenate(depthSeq, axis=0).astype(np.uint8)
    #             writer.write_image(f"depth_{imgId:04d}.png", depth_out)

    #             d = depth_out.astype(np.float32)[:, :, :-1] / 255.0
    #             l = output.astype(np.float32) / 255.0
    #             m = d.mean(axis=2) < 0.90
    #             l[m] = l[m] * 0.3 + d[m] * 0.7
    #             l = np.clip(l, 0.0, 1.0)
    #             writer.write_image(f"blend_{imgId:04d}.png", utils.to_uint8_img(l))

    #             imgId += 1
    #             imgSeq.clear()
    #             depthSeq.clear()
    #             startRow = -1
    #             cachedRow = 0

    #         imgSeq.append(utils.to_uint8_img(resized, False))
    #         depthSeq.append(clr_depth(depth_frame))
    #         startRow = starth + h

    #     cachedRow = startRow
    # if len(imgSeq) > 0:
    #     output = np.concatenate(imgSeq, axis=0).astype(np.uint8)
    #     writer.write_image(f"sequence_{imgId:04d}.png", output)
    #     depth_out = np.concatenate(depthSeq, axis=0).astype(np.uint8)
    #     writer.write_image(f"depth_{imgId:04d}.png", depth_out)

    #     d = depth_out.astype(np.float32)[:, :, :-1] / 255.0
    #     l = output.astype(np.float32) / 255.0
    #     m = d.mean(axis=2) < 0.90
    #     l[m] = l[m] * 0.3 + d[m] * 0.7
    #     l = np.clip(l, 0.0, 1.0)
    #     writer.write_image(f"blend_{imgId:04d}.png", utils.to_uint8_img(l))

    #     imgId += 1
    #     imgSeq.clear()
    #     depthSeq.clear()
