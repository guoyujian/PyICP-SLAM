
import argparse

import numpy as np
np.set_printoptions(precision=4)

from tqdm import tqdm

from utils.ScanContextManager import *

from utils.UtilsMisc import *

import open3d as o3d
import os.path as osp
from loguru import logger

from scipy.spatial.transform import Rotation as R




# params
parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=5000) # 5000 is enough for real time

parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper

parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)

parser.add_argument('--data_base_dir', type=str,
                    default='D:/kitti/dataset/sequences')
parser.add_argument('--sequence_idx', type=str, default='06')

parser.add_argument('--save_gap', type=int, default=300)

parser.add_argument('--use_open3d', action='store_true')

args = parser.parse_args()

# dataset
sequence_dir = "C:/Users/jkkc/Desktop/TEST/1_transed"

scan_paths = os.listdir(sequence_dir)

num_frames = len(scan_paths)

merged_pcd = o3d.geometry.PointCloud()

# Scan Context Manager (for loop detection) initialization
SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                                        num_candidates=args.num_candidates,
                                        threshold=args.loop_threshold)

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)

vis.add_geometry(merged_pcd)
to_reset = True

for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(osp.join(sequence_dir, scan_path))
    downsampled_pcd = pcd.random_down_sample(0.1)

    SCM.addNode(node_idx=for_idx, ptcloud=np.asarray(downsampled_pcd.points))

    # loop detection and optimize the graph
    if (for_idx > 1 and for_idx % args.try_gap_loop_detection == 0):
        # 1/ loop detection
        loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
        if loop_idx is not None:  # NOT FOUND
            loop_array = SCM.getPtcloud(loop_idx)
            loop_pcd = o3d.geometry.PointCloud()
            loop_pcd.points = o3d.utility.Vector3dVector(loop_array)

            reg_icp: o3d.pipelines.registration.RegistrationResult = o3d.pipelines.registration.registration_icp(
                downsampled_pcd,
                loop_pcd,
                .02,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            pcd.transform(reg_icp.transformation)
            logger.info(f"Loop event detected: now_frame_id: {for_idx}; loop_frame_id: {loop_idx}; min_dist: {loop_dist}; yaw_diff_deg: {yaw_diff_deg}")
            logger.info(f"{for_idx}'s pcd_name: {scan_paths[for_idx]}; {loop_idx}'s pcd_name: {scan_paths[loop_idx]};")
            rotate_m = reg_icp.transformation[:3,:3].copy()
            r = R.from_matrix(rotate_m)
            euler_angles = r.as_euler('xyz', degrees=True)
            if np.abs(euler_angles[0]) > 1 or np.abs(euler_angles[1]) > 1 or np.abs(euler_angles[2]) > 1:
                continue
            logger.info(f'euler_angles: {euler_angles}; rmse: {reg_icp.inlier_rmse}')

        merged_pcd += pcd
        vis.update_geometry(merged_pcd)

        if to_reset:
            vis.reset_view_point(True)
            to_reset = False

        vis.poll_events()
        vis.update_renderer()


vis.destroy_window()
o3d.io.write_point_cloud(f'C:/Users/jkkc/Desktop/TEST/merged_pcd_without_loop.pcd', merged_pcd)

