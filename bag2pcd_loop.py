'''
2024/6/26 GYJ
新版解析bag包代码，最终得到的pcd带有intensity值
'''

from functools import wraps
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import argparse
from gtsam import Pose3, Values, NonlinearFactorGraph, BetweenFactorPose3
from gtsam.symbol_shorthand import X
np.set_printoptions(precision=4)
from tqdm import tqdm
from utils.ScanContextManager import *
from utils.UtilsMisc import *
from loguru import logger





def timer(func):
    '''
    tool func: 计时器
    Args:
        func:

    Returns:

    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
        return res
    return wrapper


############### 全局配置区-start ###############

# odometry topic config
odometry_topic = '/lio_sam/mapping/odometry_incremental_correct'
# pcd topic config
pcd_topic = '/os_cloud_node/points'
# dtype name
dtype = np.dtype([
                   ('x', np.float32),
                   ('y', np.float32),
                   ('z', np.float32),
                   ('intensity', np.float32)
                ])
# filter distance
MIN_DISTANCE = .7
MAX_DISTANCE = 70
# subsampled points num.
KEEP_POINTS_NUM = 6000
LOOP_DETECTION = True

ODOM_OFFSET_THRESHOLD = 10

############### 全局配置区-end ###############

def count_pcd_and_ode(info):
    '''
    计算需要解析的 odom 和 pcd 文件数量
    Args:
        info:
    Returns:
    '''
    pcd_count = 0
    ode_count = 0
    if odometry_topic in info.topics:
        ode_count = info.topics[odometry_topic].message_count

    if pcd_topic in info.topics:
        pcd_count = info.topics[pcd_topic].message_count
    return pcd_count, ode_count


def msg2points(msg):
    '''
    将点数据筛选掉不合适距离后，转换为np数组
    Args:
        msg:

    Returns:

    '''
    points_data = np.array(list(pc2.read_points(msg, field_names=['x', 'y', 'z', 'intensity'])),
                           dtype=dtype)
    mask_0 = (points_data['x'] != 0) | (points_data['y'] != 0) | (points_data['z'] != 0)
    d = np.sqrt(points_data['x'] ** 2 + points_data['y'] ** 2 + points_data['z'] ** 2)

    mask_1 = (d >= MIN_DISTANCE) & (d <= MAX_DISTANCE)

    filter_points = points_data[mask_0 & mask_1]

    # filter_points['z'] += Z_AMENDMENT_VALUE
    return filter_points

def msg2transformation_matrix(msg):
    '''
    将odom数据转换为4x4变换矩阵
    Args:
        msg:

    Returns:

    '''
    ode = msg.pose.pose
    pos_x = ode.position.x
    pos_y = ode.position.y
    pos_z = ode.position.z
    ori_x = ode.orientation.x
    ori_y = ode.orientation.y
    ori_z = ode.orientation.z
    ori_w = ode.orientation.w
    translation_vector = np.array([pos_x, pos_y, pos_z])  # 定义平移向量
    quaternion = np.array([ori_x, ori_y, ori_z, ori_w])  # 定义四元数

    # 将四元数转换为旋转矩阵
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def apply_transform(frames, transform, optimized_result = None):
    '''
    将点云数据转换到优化后的位姿下，返回变换后的帧数组和总点数
    如果没有优化位姿，就不进行转换，直接返回原始帧数组和总点数
    Args:
        frames: 帧数组
        transform: 是否需要转换flag
        optimized_result: 优化位姿
    Returns:
        frames: 变换后的帧数组
        point_total_count: 所有帧加起来的点数
    '''
    print(4, flush=True, end='')
    print(f'total: {len(frames)}', flush=True, end='')
    point_total_count = 0

    if transform and optimized_result is not None:
        for i, frame in enumerate(frames):
            print(f'curr: {i}', flush=True, end='')
            optimized_pose = optimized_result.atPose3(X(i))
            optimized_transformation = np.array(optimized_pose.matrix())
            points = frame['pcd_array']

            points_homogeneous = np.column_stack((points['x'], points['y'], points['z'], np.ones(len(points))))

            transformed_points = np.dot(points_homogeneous, optimized_transformation.T)[:, :3]
            new_points = np.array(
                [(transformed_points[i, 0], transformed_points[i, 1], transformed_points[i, 2],
                  points[i]['intensity'])
                 for i in range(len(transformed_points))],
                dtype=dtype
            )
            frame['pcd_array'] = new_points
            point_total_count += new_points.shape[0]
    elif transform and optimized_result is None:
        for i, frame in enumerate(frames):
            print(f'curr: {i}', flush=True, end='')
            odom = frame['odom']

            points = frame['pcd_array']

            points_homogeneous = np.column_stack((points['x'], points['y'], points['z'], np.ones(len(points))))

            transformed_points = np.dot(points_homogeneous, odom.T)[:, :3]
            new_points = np.array(
                [(transformed_points[i, 0], transformed_points[i, 1], transformed_points[i, 2],
                  points[i]['intensity'])
                 for i in range(len(transformed_points))],
                dtype=dtype
            )
            frame['pcd_array'] = new_points
            point_total_count += new_points.shape[0]
    elif not transform:
        for i, frame in enumerate(frames):
            print(f'curr: {i}', flush=True, end='')
            point_total_count += frame['pcd_array'].shape[0]

    return frames, point_total_count


@timer
def merge_points(frames, point_total_count):
    '''
    将帧数组合并成一个点云数组
    传入正确的point_total_count会更高效
    Args:
        frames:
        point_total_count:

    Returns:
        res: 合并后的点云数组

    '''
    res = np.empty(point_total_count, dtype=dtype)
    start = 0
    for frame in frames:
        array = frame['pcd_array']
        length = len(array)
        res[start: start+length] = array
        start += length
    logger.info(f'point_total_count : {point_total_count}; start : {start}')
    return res

@timer
def save_pcd_bin(pcd_path, points):
    '''
    将点云数组写入到pcd文件中
    Args:
        pcd_path: 保存的pcd文件路径
        points: 点云数组

    Returns:

    '''
    print(5, flush=True, end='')
    num_points = points.shape[0]
    logger.info(f'write into {pcd_path}; num of points : {num_points}')
    if num_points <= 0:
        logger.error(f"[ERROR] NUM OF POINTS is {num_points}")


    with open(pcd_path, 'wb') as f:
        # write into header info
        header = b'# .PCD v0.7 - Point Cloud Data file format\n' \
                b'VERSION 0.7\n' \
                b'FIELDS x y z intensity\n' \
                b'SIZE 4 4 4 4\n' \
                b'TYPE F F F F\n' \
                b'COUNT 1 1 1 1\n' \
                b'WIDTH ' + bytes(str(num_points), 'utf-8') + b'\n' \
                b'HEIGHT 1\n' \
                b'VIEWPOINT 0 0 0 1 0 0 0\n' \
                b'POINTS ' + bytes(str(num_points), 'utf-8') + b'\n' \
                b'DATA binary\n'
        f.write(header)
        points.tofile(f)

@timer
def make_frames_list(frames_dict):
    '''
    根据字典提取帧数据，并对其进行加工，生成帧数组
    - 对帧数据进行采样
    - 计算每一帧的偏移量累加值
    Args:
        frames_dict: 帧字典，其key是时间戳，其value是一个字典，包含odom和pcd_array

    Returns:
        frames: 帧数组
    '''
    prev_offset = 0
    frames = []
    offset_accumulation = 0
    # sorted_keys = list(sorted(frames_dict))
    # for i, key in enumerate(sorted_keys):
    print(2, flush= True, end='')
    print(f'total: {len(frames_dict)}', flush= True, end='')
    for i, key in enumerate(frames_dict):
        frame = frames_dict[key]
        if frame['odom'] is None or frame['pcd_array'] is None:
            continue
        pcd_array = frame['pcd_array']
        pcd_obj = o3d.geometry.PointCloud()

        pcd_obj.points = o3d.utility.Vector3dVector(np.column_stack((pcd_array['x'], pcd_array['y'], pcd_array['z'])))

        num_points = len(pcd_obj.points)

        sampling_rate = KEEP_POINTS_NUM / num_points

        if sampling_rate < 1:
            downsampled_pcd: o3d.geometry.PointCloud = pcd_obj.random_down_sample(sampling_rate)
            frame['sampled_pcd_obj'] = downsampled_pcd
        else:
            logger.info(f'do not need sample')
            frame['sampled_pcd_obj'] = pcd_obj

        if i != 0:
            offset_accumulation += abs(np.linalg.norm(frame['odom'][:3, 3]) - prev_offset)

        frame['offset_accumulation'] = offset_accumulation
        prev_offset = np.linalg.norm(frame['odom'][:3, 3])
        frames.append(frame)
        print(f'curr: {i}', flush= True, end='')

    return frames


@timer
def extract_frames_list(frames_dict):
    '''
    根据字典提取帧数据，不加工，生成帧数组

    Args:
        frames_dict: 帧字典，其key是时间戳，其value是一个字典，包含odom和pcd_array

    Returns:
        frames: 帧数组
    '''
    print(2, flush=True, end='')
    print(f'total: {len(frames_dict)}', flush=True, end='')
    frames = []
    # sorted_keys = list(sorted(frames_dict))
    # for i, key in enumerate(sorted_keys):
    for i, key in enumerate(frames_dict):
        frame = frames_dict[key]
        if frame['pcd_array'] is None:
            continue

        frames.append(frame)
        print(f'curr: {i}', flush=True, end='')
    return frames

@timer
def loop_detection_and_update_odom(frames):
    '''
    回环检测
    Args:
        frames:

    Returns:
        res: 返回值保存了更新后的位姿
    '''
    print(3, flush=True, end='')
    print(f'total: {len(frames)}', flush=True, end='')
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
    for frame_idx, curr_frame in enumerate(frames):
        print(f'curr: {frame_idx}', flush=True, end='')
        curr_pose = Pose3(curr_frame['odom'])
        curr_sampled_pcd = curr_frame['sampled_pcd_obj']
        curr_offset_accumulation = curr_frame['offset_accumulation']

        # 添加初始估计
        initial_estimate.insert(X(frame_idx), curr_pose)

        # if key_frame:
        curr_sampled_pcd_: o3d.geometry.PointCloud = copy.deepcopy(curr_sampled_pcd)
        curr_sampled_pcd_.transform(curr_frame['odom'])

        SCM.addNode(node_idx= frame_idx, ptcloud= np.asarray(curr_sampled_pcd_.points))

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
                    loop_sampled_pcd = frames[loop_idx]['sampled_pcd_obj']
                    loop_offset_accumulation = frames[loop_idx]['offset_accumulation']
                    # if abs(loop_offset_accumulation- curr_offset_accumulation) < 10:
                    #     continue
                    logger.info(
                        f"Loop event detected: now_frame: {frame_idx}; loop_frame: {loop_idx}; min_dist: {loop_dist}; yaw: {yaw_diff_deg}; odom_diff: {curr_offset_accumulation - loop_offset_accumulation}")
                    if abs(curr_offset_accumulation - loop_offset_accumulation) <= ODOM_OFFSET_THRESHOLD:
                        continue
                    loop_closure_relative_pose = initial_estimate.atPose3(X(loop_idx)).between(curr_pose)
                    loop_timestamp = frames[loop_idx]['timestamp']
                    curr_timestamp = frames[frame_idx]['timestamp']
                    relative_transform = np.array(loop_closure_relative_pose.matrix())
                    # logger.info(f'loop_timestamp: {loop_timestamp}; curr_frame_timestamp:{curr_timestamp}; relative: {relative_transfor}')
                    reg_icp = o3d.pipelines.registration.registration_icp(
                        source=curr_sampled_pcd,
                        target=loop_sampled_pcd,
                        max_correspondence_distance=10,
                        init=relative_transform,
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
    return optimizer.optimize()



@timer
def main(bag_file_path, pcd_file_path):
    '''


    Args:
        bag_file_path: bag文件路径
        pcd_file_path: pcd文件保存路径

    Returns:

    '''

    print(1, flush=True, end="") # 请不要删除修改这句话，他会向调用者传递开始解析的消息

    # 读取bag文件
    with rosbag.Bag(bag_file_path, 'r') as bag:
        # 读取bag信息
        info = bag.get_type_and_topic_info(topic_filters=[odometry_topic, pcd_topic])
        logger.info(f'info: {info}')
        # 计算解析文件的总数，以确定进度
        pcd_count, ode_count = count_pcd_and_ode(info)
        total_count = pcd_count + ode_count
        points_need_transform = False if ode_count == 0 else True

        logger.info(f'pcd_topic: {pcd_topic}, count: {pcd_count}')
        logger.info(f'ode_topic: {odometry_topic}, count: {ode_count}')
        logger.info(f'need transform: {points_need_transform}')

        if total_count == 0:
            logger.error(f'{bag_file_path}  文件数据无法获取')
            raise RuntimeError(f'{bag_file_path}  文件数据无法获取')

        print(f'total: {total_count+1}', flush=True, end='')

        frames_dict = {}

        cur = 0
        point_total_count = 0
        for topic, msg, t in bag.read_messages(topics=[odometry_topic, pcd_topic]):
            timestamp_key = round(msg.header.stamp.to_sec(), 3)

            if timestamp_key in frames_dict:
                frame = frames_dict[timestamp_key]
            else:
                frame = {
                    'timestamp': timestamp_key,
                    'odom': None,
                    'pcd_array': None,
                    'sampled_pcd_obj': None,
                    'offset_accumulation': 0,
                }

            if topic == pcd_topic:
                points_array = msg2points(msg)
                frame['pcd_array'] = points_array

                cur += 1
                progress = "%.2f" % (cur / total_count * 100)
                logger.info(f'parse type: points, parse time: {timestamp_key}, parse progress：{progress}%')
            elif topic == odometry_topic:
                frame['odom'] = msg2transformation_matrix(msg)

                cur += 1
                progress = "%.2f" % (cur / total_count * 100)
                logger.info(f'parse type: ode, parse time: {timestamp_key}, parse progress：{progress}%')

            frames_dict[timestamp_key] = frame

            print(f'curr: {cur}', flush=True, end='')

        if points_need_transform and LOOP_DETECTION:
            frames = make_frames_list(frames_dict)
        else:
            frames = extract_frames_list(frames_dict)
        logger.info(f'maked frames count: {len(frames)}')
        assert len(frames) > 1



        if LOOP_DETECTION and points_need_transform:

            optimized_result = loop_detection_and_update_odom(frames)
            frames, point_total_count = apply_transform(frames, True, optimized_result = optimized_result)
        elif not LOOP_DETECTION and points_need_transform:
            frames, point_total_count = apply_transform(frames, True, optimized_result = None)
        else:
            frames, point_total_count = apply_transform(frames, False, optimized_result = None)
        save_pcd_bin(pcd_file_path, merge_points(frames, point_total_count))
        print(f'curr: {total_count+1}', flush=True, end='')


def repair_unindex_bag(bag_file_path):
    '''
    修复未索引的bag文件，使其可以被正常读取
    Args:
        bag_file_path:

    Returns:

    '''
    try:
        with rosbag.Bag(bag_file_path, "r", allow_unindexed=False) as bag:
            pass
    except rosbag.bag.ROSBagUnindexedException as e:
        print(0, flush=True, end="") # 请不要删除修改这句话，他会向调用者传递bag file need to reindex的消息
        with rosbag.Bag(bag_file_path, "a", allow_unindexed=True) as bag1:
            for offset in bag1.reindex():
                pass






if __name__ == '__main__':
    # python bag2pcd_loop.py "C:\Users\jkkc\Desktop\TEST\1.bag" "C:\Users\jkkc\Desktop\TEST\1" "C:\Users\jkkc\Desktop\TEST\icp-slam.pcd"
    parser = argparse.ArgumentParser()

    parser.add_argument('bag_file_path')  # 待解析的bag
    parser.add_argument('pcd_dir_path')   # 临时文件夹
    parser.add_argument('pcd_file_path')  # 解算后的pcd位置


    # 降采样剩余的点数，用于icp和loop detection
    parser.add_argument('--num_icp_points', type=int, default=6000)  # 5000 is enough for real time
    # sacn context 的参数
    parser.add_argument('--num_rings', type=int, default=20)  # same as the original paper
    parser.add_argument('--num_sectors', type=int, default=60)  # same as the original paper
    parser.add_argument('--num_candidates', type=int, default=10)  # must be int
    parser.add_argument('--try_gap_loop_detection', type=int, default=10)  # same as the original paper

    parser.add_argument('--loop_threshold', type=float,
                        default=0.11)  # 0.11 is usually safe (for avoiding false loop closure)

    parser.add_argument('--save_gap', type=int, default=300)

    args = parser.parse_args()

    repair_unindex_bag(args.bag_file_path)

    main(args.bag_file_path, args.pcd_file_path)


