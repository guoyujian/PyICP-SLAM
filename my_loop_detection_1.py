import os
import sys
import csv
import copy
import time
import random
import argparse

import numpy as np
np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

from utils.ScanContextManager import *
from utils.PoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.ICP as ICP
import open3d as o3d
import os.path as osp
from loguru import logger



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

scan_paths = [osp.join(sequence_dir, f) for f in os.listdir(sequence_dir)]

num_frames = len(scan_paths)

# Scan Context Manager (for loop detection) initialization
SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                                        num_candidates=args.num_candidates,
                                        threshold=args.loop_threshold)



for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=5.0):

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(scan_path)
    downsampled_pcd = pcd.random_down_sample(0.1)

    # logger.info(f'downsampled pcd num: {len(downsampled_pcd.points)}')

    SCM.addNode(node_idx=for_idx, ptcloud=np.asarray(downsampled_pcd.points))


    # loop detection and optimize the graph
    if (for_idx > 1 and for_idx % args.try_gap_loop_detection == 0):
        # 1/ loop detection
        loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
        if (loop_idx == None):  # NOT FOUND
            pass
        else:
            logger.info(f"Loop event detected: now_frame_id: {for_idx}; loop_frame_id: {loop_idx}; min_dist: {loop_dist}; yaw_diff_deg: {yaw_diff_deg}")
            logger.info(f"{for_idx}'s pcd_name: {scan_paths[for_idx]}; {loop_idx}'s pcd_name: {scan_paths[loop_idx]}; ")




