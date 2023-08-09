import copy
import numpy as np
import open3d as o3
from probreg import cpd

# load source and target point cloud
source = o3.io.read_point_cloud("cloud_bin_0.pcd")
source.remove_non_finite_points()
# target = copy.deepcopy(source)
# # transform target point cloud
# th = np.deg2rad(30.0)
# target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
#                            [np.sin(th), np.cos(th), 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]]))
target = o3.io.read_point_cloud("cloud_bin_1.pcd")
source = source.voxel_down_sample(voxel_size=0.05)
target = target.voxel_down_sample(voxel_size=0.05)

# compute cpd registration
tf_param, _, _ = cpd.registration_cpd(source, target)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([source, target, result])
