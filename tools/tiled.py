import argparse

import cv2
import os
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", help="root output folder")


def main():
    args = parser.parse_args()

    splitted_path = os.path.join(args.path, "linescan", "splitted")

    splitted_images = sorted(glob.glob(os.path.join(splitted_path, "*.png")))

    num_splitted_images = len(splitted_images)

    if num_splitted_images <= 1:
        return

    tiled_path = os.path.join(args.path, "linescan", "tiled", "splitted")

    if not os.path.exists(tiled_path):
        os.makedirs(tiled_path)

    prev_img = cv2.imread(splitted_images[0], cv2.IMREAD_UNCHANGED)
    for i in range(1, num_splitted_images):
        cur_img = cv2.imread(splitted_images[i], cv2.IMREAD_UNCHANGED)

        ph, pw = prev_img.shape[:2]

        ch, cw = cur_img.shape[:2]

        tiled = np.concatenate([prev_img[ph // 2 :], cur_img[: ch // 2]], axis=0)

        cv2.imwrite(os.path.join(tiled_path, f"{i-1:04d}.png"), tiled)

        prev_img = cur_img


if __name__ == "__main__":
    main()
