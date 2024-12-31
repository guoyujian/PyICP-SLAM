import os
import os.path as osp
from glob import glob
from os import times

import numpy as np
import yaml
from jinja2.optimizer import optimize
from scipy.spatial.transform import Rotation as R

import open3d as o3d

import argparse
import copy


from gtsam import Pose3, Values, NonlinearFactorGraph, BetweenFactorPose3
from gtsam.symbol_shorthand import X


np.set_printoptions(precision=4)

from tqdm import tqdm

from utils.ScanContextManager import *
from utils.MyPoseGraphManager import *
from utils.UtilsMisc import *


from loguru import logger

from math import sqrt, pow





parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

# 降采样剩余的点数，用于icp和loop detection
parser.add_argument('--num_icp_points', type=int, default=6000) # 5000 is enough for real time
# sacn context 的参数
parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper

parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)

parser.add_argument('--save_gap', type=int, default=300)


# 包含pcd和odometry的文件夹
parser.add_argument('--pcds_dir', type=str, default='C:/Users/jkkc/Desktop/TEST/1')


args = parser.parse_args()

def get_transformation(odometry_file_path):
    '''
    辅助函数：从txt文件中读取平移向量xyz和旋转四元数
    返回平移向量和旋转矩阵
    '''
    with open(odometry_file_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    pos_x = result['pose']['pose']['position']['x']
    pos_y = result['pose']['pose']['position']['y']
    pos_z = result['pose']['pose']['position']['z']
    ori_x = result['pose']['pose']['orientation']['x']
    ori_y = result['pose']['pose']['orientation']['y']
    ori_z = result['pose']['pose']['orientation']['z']
    ori_w = result['pose']['pose']['orientation']['w']
    # 定义平移向量
    translation_vector = np.array([pos_x, pos_y, pos_z])
    quaternion = np.array([ori_x, ori_y, ori_z, ori_w])  # 定义四元数

    # 将四元数转换为旋转矩阵
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation_vector
    return transformation


def getPointCloud(pcd_file, transformation):
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcd_file)

    # 过滤小于最小值大于最大值的点
    points = np.asarray(pcd.points)
    # 计算每个点到坐标原点的L2范数
    distances = np.linalg.norm(points, axis=1)
    # 根据索引保留点
    pcd = pcd.select_by_index(
        np.where((distances >= 0.7) & (distances <= 70))[0])

    keep_points_num = 6000

    num_points = len(pcd.points)

    sampling_rate = keep_points_num / num_points

    if sampling_rate < 1:
        # logger.info(f'random sampling: {sampling_rate}')
        downsampled_pcd: o3d.geometry.PointCloud = pcd.random_down_sample(sampling_rate)
        # downsampled_pcd.transform(transformation)
        return pcd, downsampled_pcd
    else:
        logger.info(f'do not need sample')
        transed_pcd = copy.deepcopy(pcd)
        return pcd, transed_pcd



def get_frames(pcd_files):
    frames = []
    for i, pcd_f in enumerate(pcd_files):
        timestamp = float(osp.basename(pcd_f)[:-4])
        transformation = get_transformation(pcd_f[:-4] + '.txt')
        pcd_obj, downsampled_pcd_obj = getPointCloud(pcd_f, transformation)
        frame = {
            'timestamp': timestamp,
            'pos': transformation,
            'pcd': pcd_obj,
            'sampled_pcd': downsampled_pcd_obj,
        }
        frames.append(frame)

    return frames



def main():


    pcd_files = glob(osp.join(args.pcds_dir, '*.pcd'))

    pcd_files = [pcd_file for pcd_file in pcd_files if osp.exists(pcd_file[:-4] + '.txt')]

    logger.info(f'found pcd files num : {len(pcd_files)}')

    frames = get_frames(pcd_files)
    merged_pcd = o3d.geometry.PointCloud()


    num_frames = len(frames)

    logger.info(f'maked frames count: {num_frames}')

    assert num_frames > 1


    # Scan Context Manager (for loop detection) initialization
    SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                             num_candidates=args.num_candidates,
                             threshold=args.loop_threshold)

    # 定义噪声模型
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))  # XYZRPY
    loop_closure_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))

    # 初始化因子图和初始估计
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    # 遍历每一帧数据

    for frame_idx, curr_frame in tqdm(enumerate(frames), total=num_frames, mininterval=5.0):
        curr_pose = Pose3(curr_frame['pos'])
        curr_sampled_pcd = curr_frame['sampled_pcd']

        # 添加初始估计
        initial_estimate.insert(X(frame_idx), curr_pose)

        # if key_frame:
        SCM.addNode(node_idx= frame_idx, ptcloud= np.asarray(curr_sampled_pcd.points))

        if frame_idx > 0:
            # 获取上一帧的位姿
            prev_pose = initial_estimate.atPose3(X(frame_idx - 1))

            # 计算相对位姿作为 odometry 约束
            relative_pose = prev_pose.between(curr_pose)

            # 添加 odometry 因子
            graph.add(BetweenFactorPose3(X(frame_idx - 1), X(frame_idx), relative_pose, odometry_noise))
            if frame_idx % args.try_gap_loop_detection == 0:
                # loop detection
                loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
                if loop_idx is not None:  # NOT FOUND
                    loop_sampled_pcd = frames[loop_idx]['sampled_pcd']
                    logger.info(
                        f"Loop event detected: now_frame_id: {frame_idx}; loop_frame_id: {loop_idx}; min_dist: {loop_dist}; yaw_diff_deg: {yaw_diff_deg}")
                    loop_closure_relative_pose = initial_estimate.atPose3(X(loop_idx)).between(curr_pose)
                    loop_timestamp = frames[loop_idx]['timestamp']
                    curr_timestamp = frames[frame_idx]['timestamp']
                    relative_transfor = np.array(loop_closure_relative_pose.matrix())
                    # logger.info(f'loop_timestamp: {loop_timestamp}; curr_frame_timestamp:{curr_timestamp}; relative: {relative_transfor}')
                    reg_icp = o3d.pipelines.registration.registration_icp(
                        source=curr_sampled_pcd,
                        target=loop_sampled_pcd,
                        max_correspondence_distance=10,
                        init=relative_transfor,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
                    )

                    rotate_m = reg_icp.transformation[:3, :3].copy()
                    r = R.from_matrix(rotate_m)
                    euler_angles = r.as_euler('xyz', degrees=True)
                    if np.abs(euler_angles[0]) > 1 or np.abs(euler_angles[1]) > 1 or np.abs(euler_angles[2]) > 1:
                        continue
                    odom_transform = np.asarray(reg_icp.transformation)
                    graph.add(BetweenFactorPose3(X(loop_idx), X(frame_idx), Pose3(odom_transform), loop_closure_noise))



    # 优化因子图
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()


    # 提取优化后的位姿
    with open(r'C:\Users\jkkc\Desktop\TEST\trans.txt', 'w', encoding='UTF-8') as f:
        for i in range(num_frames):
            optimized_pose = result.atPose3(X(i))
            optimized_transformation = np.array(optimized_pose.matrix())
            pcd = frames[i]['pcd']
            transformation = frames[i]['pos']
            str_trans = np.array2string(transformation).replace('\n', '')
            opt_str_trans = np.array2string(optimized_transformation).replace('\n', '')
            f.write(f'---- {i} ----\n{str_trans}\n{opt_str_trans}\n')
            pcd.transform(optimized_transformation)
            merged_pcd += pcd

    o3d.io.write_point_cloud(f'C:/Users/jkkc/Desktop/TEST/icp_slam.pcd',
         merged_pcd
    )


if __name__ == '__main__':
    main()