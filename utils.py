import numpy as np
import os
import cv2
import open3d as o3d
from skimage import morphology
from matplotlib import pyplot as plt

from common import common


def get_content(lines, line_start, line_end):
    if line_start is not None and line_end is not None:
        slices = lines[line_start:line_end]

    elif line_start is not None:
        slices = lines[line_start:]

    elif line_end is not None:
        slices = lines[:line_end]

    else:
        slices = lines

    return "\n".join(slices)


def calc_iou(template, sample):
    mask_template = template > 0
    mask_sample = sample > 0
    mask_union = mask_template | mask_sample
    mask_intersect = mask_template & mask_sample
    if mask_union.sum() == 0:
        return 0.0
    return mask_intersect.sum() / mask_union.sum()


def calc_iou_template(template, sample):
    mask_template = template > 0
    mask_sample = sample > 0
    mask_intersect = mask_template & mask_sample
    if mask_template.sum() == 0:
        return 0.0
    return mask_intersect.sum() / mask_template.sum()


def normalize(data):
    minv = data.min()
    maxv = data.max()

    if maxv == np.inf:
        new_data = data.copy()
        new_data[data == np.inf] = 1.0
        new_data[data < np.inf] = 0.0
        return new_data
    elif minv == -np.inf:
        new_data = data.copy()
        new_data[data == -np.inf] = 0.0
        new_data[data > -np.inf] = 1.0
        return new_data

    new_data = (data - minv) / (maxv - minv + 1e-10)

    return new_data


def fit_in_range(data, minv, maxv):
    assert minv < maxv, "min value should be less than max value!"

    new_data = data.copy()

    if maxv == np.inf:
        new_data[data == np.inf] = 1.0
        new_data[data < np.inf] = 0.0
        return new_data
    elif minv == -np.inf:
        new_data[data == -np.inf] = 0.0
        new_data[data > -np.inf] = 1.0
        return new_data

    new_data[new_data < minv] = minv
    new_data[new_data > maxv] = maxv

    new_data = (new_data - minv) / (maxv - minv + 1e-10)

    return new_data


def to_uint8_img(image, require_normalize=True):
    return (
        (normalize(image.astype(np.float32)) * 255).astype(np.uint8)
        if require_normalize
        else (image.astype(np.float32) * 255).astype(np.uint8)
    )


def get_canny_edges(image, low_thresh=85, high_thresh=255):
    return cv2.Canny(to_uint8_img(image), low_thresh, high_thresh)


def pyramid(image, level=2):
    img = to_uint8_img(image.copy())

    for _ in range(level):
        img = cv2.pyrDown(img)

    for _ in range(level):
        img = cv2.pyrUp(img)

    return img


def merge_pcd(pcds):
    pts = [np.asarray(pcd.points) for pcd in pcds]
    # clrs = [np.asarray(pcd.colors) for pcd in pcds]
    merged_pts = np.concatenate(pts, axis=0)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(merged_pts)

    return pcd


def visualize_pcds(pcds):
    o3d.visualization.draw_geometries(pcds)


def export_pcd(export_path, pcd):
    o3d.io.write_point_cloud(export_path, pcd)


import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source, target):
    trans_init = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold
    )
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     3, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
    #             0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    #             distance_threshold)
    #     ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def remove_noise(gray, num):
    Y, X = gray.shape
    nearest_neigbours = [
        [
            np.argmax(
                np.bincount(
                    gray[
                        max(i - num, 0) : min(i + num, Y),
                        max(j - num, 0) : min(j + num, X),
                    ].ravel()
                )
            )
            for j in range(X)
        ]
        for i in range(Y)
    ]
    result = np.array(nearest_neigbours, dtype=np.uint8)
    # cv2.imwrite('result2.jpg', result)
    return result


def remove_small_objects(arr, min_size):
    if arr.dtype == np.uint8:
        im = arr == 255
    elif arr.dtype == bool:
        im = arr
    else:
        assert False
    cleaned = morphology.remove_small_objects(im, min_size)
    return to_uint8_img(cleaned)


def remove_small_holes(arr, area_threshold):
    if arr.dtype == np.uint8:
        im = arr == 255
    elif arr.dtype == bool:
        im = arr
    else:
        assert False
    cleaned = morphology.remove_small_holes(im, area_threshold)
    return to_uint8_img(cleaned)


def get_clean_mask(mask, min_obj_size, min_hole_size):
    clean_mask = remove_small_objects(mask, min_obj_size)
    clean_mask = remove_small_holes(clean_mask, min_hole_size)
    return clean_mask


def get_hls_mask(image):
    if image.dtype != np.uint8:
        image_copy = to_uint8_img(image, require_normalize=False)
    else:
        image_copy = image.copy()

    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)

    lightness = hls[:, :, 1]

    _, thresh = cv2.threshold(lightness, 80, 255, cv2.THRESH_BINARY)

    h, w = thresh.shape
    min_obj_size = (min(h, w) // 100) ** 2
    min_hole_size = 100

    hls_mask = get_clean_mask(thresh, min_obj_size, min_hole_size)

    return hls_mask


def get_rgba(val, cmap="rainbow"):
    assert 0 <= np.any(val) <= 1
    return np.array(plt.cm.get_cmap(cmap)(val))


def visualize(image, export_path=None):
    plt.clf()
    if export_path is None:
        plt.imshow(image) if len(image.shape) == 3 else plt.imshow(image, cmap="gray")
        plt.show()
    else:
        out_path = os.path.join(common.output_path, export_path)
        plt.imsave(out_path, image) if len(image.shape) == 3 else plt.imsave(
            out_path, image, cmap="gray"
        )

    return image
