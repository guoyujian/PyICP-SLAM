import os
import os.path as osp
from glob import glob
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

import open3d as o3d

import argparse
import copy

np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

from utils.ScanContextManager import *
from utils.MyPoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.ICP as ICP

from loguru import logger



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
        logger.info(f'random sampling: {sampling_rate}')
        downsampled_pcd: o3d.geometry.PointCloud = pcd.random_down_sample(sampling_rate)
        downsampled_pcd.transform(transformation)
        return pcd, downsampled_pcd
    else:
        logger.info(f'do not need sample')
        transed_pcd = copy.deepcopy(pcd)
        return pcd, transed_pcd




def main():
    pcd_files = glob(osp.join(args.pcds_dir, '*.pcd'))

    pcd_files = [pcd_file for pcd_file in pcd_files if osp.exists(pcd_file[:-4] + '.txt')]

    logger.info(f'found pcd files num : {len(pcd_files)}')

    frames = []
    merged_pcd = o3d.geometry.PointCloud()

    for pcd_f in pcd_files:
        timestamp = float(osp.basename(pcd_f)[:-4])
        transformation = get_transformation(pcd_f[:-4] + '.txt')
        pcd_obj, downsampled_pcd_obj = getPointCloud(pcd_f, transformation)
        frame = {
            'timestamp': timestamp,
            'pos': transformation,
            'pcd': pcd_obj,
            'sampled_pcd': downsampled_pcd_obj
        }
        frames.append(frame)

    num_frames = len(frames)

    logger.info(f'maked frames count: {num_frames}')

    assert num_frames > 1

    # Pose Graph Manager (for back-end optimization) initialization
    PGM = PoseGraphManager()

    # Scan Context Manager (for loop detection) initialization
    SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                             num_candidates=args.num_candidates,
                             threshold=args.loop_threshold)

    correct_transformation = np.eye(4)

    for frame_idx, curr_frame in tqdm(enumerate(frames), total=num_frames, mininterval=5.0):


        curr_timestamp = curr_frame['timestamp']
        curr_pos = curr_frame['pos']
        curr_pcd = curr_frame['pcd']
        curr_sampled_pcd = curr_frame['sampled_pcd']

        SCM.addNode(node_idx= frame_idx, ptcloud= np.asarray(curr_sampled_pcd.points))

        if frame_idx == 0:
            # 初始化
            logger.info('init pgm')
            PGM.addPriorFactor(curr_pos)
            continue

        prev_frame = frames[frame_idx - 1]

        prev_timestamp = prev_frame['timestamp']
        prev_pos = prev_frame['pos']
        prev_pcd = prev_frame['pcd']
        prev_sampled_pcd = prev_frame['sampled_pcd']

        PGM.prev_node_idx = frame_idx - 1
        PGM.curr_node_idx = frame_idx

        PGM.curr_se3 = curr_pos

        prev_curr_transformation = np.matmul(
            np.linalg.inv(prev_pos),
            curr_pos
        )

        PGM.addOdometryFactor(prev_curr_transformation)


        # loop detection and optimize the graph
        if PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0:

            # loop detection
            loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
            if loop_idx is not None:  # NOT FOUND
                loop_frame = frames[loop_idx]
                loop_timestamp = loop_frame['timestamp']
                loop_pos = loop_frame['pos']
                loop_pcd = loop_frame['pcd']
                loop_sampled_pcd = loop_frame['sampled_pcd']

                reg_icp: o3d.pipelines.registration.RegistrationResult = o3d.pipelines.registration.registration_icp(
                    curr_sampled_pcd,
                    loop_sampled_pcd,
                    .02,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )

                # 评估icp的结果是否符合
                logger.info(
                    f"Loop event detected: now_frame_id: {frame_idx}; loop_frame_id: {loop_idx}; min_dist: {loop_dist}; yaw_diff_deg: {yaw_diff_deg}")
                logger.info(f"{frame_idx}'s pcd_name: {curr_timestamp}; {loop_idx}'s pcd_name: {loop_timestamp};")
                rotate_m = reg_icp.transformation[:3, :3].copy()
                r = R.from_matrix(rotate_m)
                euler_angles = r.as_euler('xyz', degrees=True)
                if np.abs(euler_angles[0]) > 1 or np.abs(euler_angles[1]) > 1 or np.abs(euler_angles[2]) > 1:
                    continue

                # # update 后续所有frame的pose
                # for f in frames[frame_idx:]:
                #     f['pos'] = np.matmul(f['pos'], reg_icp.transformation)

                # PGM.addLoopFactor(reg_icp.transformation.copy(), loop_idx)
                #
                # # correct
                # PGM.optimizePoseGraph()
                correct_transformation = np.matmul(correct_transformation, reg_icp.transformation)
        new_curr_pos = np.matmul(curr_pos, correct_transformation)
        curr_pcd.transform(new_curr_pos)
        merged_pcd+=curr_pcd





    # merged_pcd = o3d.geometry.PointCloud()

    # for opt_idx in range(num_frames):
    #     pose_trans, pose_rot = getGraphNodePose(PGM.graph_optimized, opt_idx)
    #     frame = frames[opt_idx]
    #     correct_pos = np.eye(4)
    #     correct_pos[:3, :3] = pose_rot
    #     correct_pos[:3, 3] = pose_trans
    #     # frame['pos'] = correct_pos
    #     pcd:o3d.geometry.PointCloud = frame['pcd']
    #     pcd.transform(correct_pos)
    #     merged_pcd += pcd

    o3d.io.write_point_cloud(f'C:/Users/jkkc/Desktop/TEST/icp_slam.pcd',
         merged_pcd
    )




if __name__ == '__main__':
    main()
