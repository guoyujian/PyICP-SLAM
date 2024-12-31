'''
2024/6/26 GYJ
新版本解析bag包代码，最终得到的pcd带有intensity值
'''
import sys

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
from loguru import logger
import argparse
from scipy.spatial.transform import Rotation as R
from sklearn.utils import deprecated
from tqdm import tqdm
from functools import wraps
import time

from win32trace import flush

import os
import os.path as osp
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

from loguru import logger
import open3d as o3d
import argparse
from glob import glob
from merge_pointcloud import SimpleMerge, SimpleMergeWithoutOdometer
from my_loop_detection import loop_and_merge


# 计时器
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
        return res
    return wrapper



# odometry和pcd的topic写死
odometry_topic = '/lio_sam/mapping/odometry_incremental_correct'
# odometry_topic = '/lio_sam/mapping/odometry'
pcd_topic = '/os_cloud_node/points'
dtype = np.dtype([
                   ('x', np.float32),
                   ('y', np.float32),
                   ('z', np.float32),
                   ('intensity', np.float32)
                ])
MIN_DISTANCE = .7
MAX_DISTANCE = 700

def count_pcd_and_ode(info):
    '''
    计算需要解析的文件总数
    '''
    pcd_count = 0
    ode_count = 0
    if odometry_topic in info.topics:
        ode_count = info.topics[odometry_topic].message_count

    if pcd_topic in info.topics:
        pcd_count = info.topics[pcd_topic].message_count
    return pcd_count, ode_count


def msg2points(msg):
    points_data = np.array(list(pc2.read_points(msg, field_names=['x', 'y', 'z', 'intensity'])),
                           dtype=dtype)
    mask_0 = (points_data['x'] != 0) | (points_data['y'] != 0) | (points_data['z'] != 0)
    d = np.sqrt(points_data['x'] ** 2 + points_data['y'] ** 2 + points_data['z'] ** 2)

    mask_1 = (d >= MIN_DISTANCE) & (d <= MAX_DISTANCE)

    filter_points = points_data[mask_0 & mask_1]

    # filter_points['z'] += Z_AMENDMENT_VALUE
    return filter_points

def msg2transformation_matrix(msg):
    """
    获取Odometry
    """
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


def apply_transform(pcd_dict: dict, ode_dict: dict):
    '''
    应用变换
    '''
    new_pcd_dict = {}
    logger.info(f'There are total {len(pcd_dict)} pcds.')

    pcd_dict = {key: val for key, val in pcd_dict.items() if key in ode_dict}

    len_pcd_dict = len(pcd_dict)

    logger.info(f'There are total {len(pcd_dict)} pcds matching odometry.')
    point_total_count = 0
    for idx, key in tqdm(enumerate(pcd_dict), desc="Starting to transform points according to odometry: ", disable=True):
        points = pcd_dict[key]
        transformation_matrix = ode_dict[key]

        points_homogeneous = np.column_stack((points['x'], points['y'], points['z'], np.ones(len(points))))
        transformed_points = np.dot(points_homogeneous, transformation_matrix.T)[:, :3]

        new_points = np.array(
            [(transformed_points[i, 0], transformed_points[i, 1], transformed_points[i, 2],
              points[i]['intensity'])
            for i in range(len(transformed_points))],
            dtype=dtype
        )
        new_pcd_dict[key] = new_points
        point_total_count += new_points.shape[0]

        # print(idx * 100 / len_pcd_dict)
    return new_pcd_dict, point_total_count

@timer
def merge_points(pcd_dict, point_total_count):
    res = np.empty(point_total_count, dtype=dtype)
    start = 0
    for key in pcd_dict:
        array = pcd_dict[key]
        length = len(array)
        res[start: start+length] = array
        start += length
    logger.info(f'point_total_count : {point_total_count}; start : {start}')
    return res


def save_pcd_bin(pcd_path, points):
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
def main(bag_file_path, pcd_file_path):

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

        pcd_dict = {}
        ode_dict = {}

        cur = 0
        point_total_count = 0
        for topic, msg, t in bag.read_messages(topics=[odometry_topic, pcd_topic]):
            time_str = "%.3f" % msg.header.stamp.to_sec()
            if topic == pcd_topic:
                points_array = msg2points(msg)
                pcd_dict[time_str] = points_array
                point_total_count += points_array.shape[0]

                cur += 1
                progress = "%.2f" % (cur / total_count * 100)
                logger.info(f'parse type: points, parse time: {time_str}, parse progress：{progress}%')
            elif topic == odometry_topic:
                ode_dict[time_str] = msg2transformation_matrix(msg)

                cur += 1
                progress = "%.2f" % (cur / total_count * 100)
                logger.info(f'parse type: ode, parse time: {time_str}, parse progress：{progress}%')
            print(f'curr: {cur}', flush=True, end='')

        logger.info(f'need transform points: {points_need_transform}')
        if points_need_transform:
            pcd_dict, point_total_count = apply_transform(pcd_dict, ode_dict)
        save_pcd_bin(pcd_file_path, merge_points(pcd_dict, point_total_count))
        print(f'curr: {total_count+1}', flush=True, end='')


def repair_unindex_bag(bag_file_path):
    try:
        with rosbag.Bag(bag_file_path, "r", allow_unindexed=False) as bag:
            pass
    except rosbag.bag.ROSBagUnindexedException as e:
        print(1, flush=True, end="") # 请不要删除修改这句话，他会向调用者传递bag file need to reindex的消息
        with rosbag.Bag(bag_file_path, "a", allow_unindexed=True) as bag1:
            for offset in bag1.reindex():
                pass
    print(2, flush=True, end="") # 请不要删除修改这句话，他会向调用者传递start to merge的消息
    print("", flush=True) # 请不要删除修改这句话



class BagExtractor:

    def __init__(self, bag_file, dst_folder):
        self.bag_file = bag_file
        self.dst_folder = dst_folder
        self.bridge = CvBridge()

    def run(self):
        """
        读取bag文件
        :return:
        """
        # 读取bag文件
        with rosbag.Bag(self.bag_file, 'r') as bag:
            # 读取bag信息

            info = bag.get_type_and_topic_info()


            # 计算解析文件的总数，以确定进度
            total = self.total_count(info)
            # 读取pcd坐标的topic
            # pcd_topic, odometry_topic, total = self.read_bag_point_topic(info)
            logger.info("pcd_topic: {}, odometry_topic: {}, total: {}".format(pcd_topic, odometry_topic, total))
            if total == 0:
                logger.error(f'{self.bag_file}  文件数据无法获取')
                raise RuntimeError(f'{self.bag_file}  文件数据无法获取')
            # 读取信息
            cur = 0
            for topic, msg, t in bag.read_messages():
                # print(type(t))
                if topic == pcd_topic:
                    # 读取时间戳
                    time_str = "%.3f" % msg.header.stamp.to_sec()
                    # 文件地址
                    pcd_path = os.path.join(self.dst_folder, "{}.pcd".format(time_str))
                    cur += 1
                    progress = "%.2f" % (cur / total * 100)
                    # 转ascii码 生成文件
                    self.to_pcd(pcd_path, msg)
                    logger.info(f'文件总数：{total}, 生成第{cur}个文件：{pcd_path}, 进度：{progress}%')
                elif topic == odometry_topic:
                    # time = msg.header.stamp.secs + msg.header.stamp.nsecs * (10 ** -9)
                    # 读取时间戳
                    # time_str = "%.9f" % time

                    time_str = "%.3f" % msg.header.stamp.to_sec()
                    logger.info("时间戳：{}".format(time_str))
                    cur += 1
                    progress = "%.2f" % (cur / total * 100)
                    # 文件地址
                    txt_path = os.path.join(self.dst_folder, "{}.txt".format(time_str))
                    self.to_txt_ascii(txt_path, msg)
                    logger.info(f'文件总数：{total}, 生成第{cur}个文件: {txt_path}, 进度：{progress}%')

    def total_count(self, info):
        '''
        计算需要解析的文件总数
        :param info:
        :return:
        '''
        total = 0

        if odometry_topic in info.topics:
            total += info.topics[odometry_topic].message_count

        if pcd_topic in info.topics:
            total += info.topics[pcd_topic].message_count

        return total

    @staticmethod
    def to_txt_ascii(txt_path, msg):
        """
        获取Odometry, 生成文件
        :param txt_path:
        :param msg:
        :return:
        """
        # logger.info(f'{msg}')
        with open(txt_path, mode='w', encoding='gbk') as file:
            file.write(str(msg))



    @staticmethod
    def to_pcd(pcd_path, msg):
        """
        获取坐标，利用open3d生成文件
        :param pcd_path:
        :param msg:
        :return:
        """
        points_data = list(pc2.read_points(msg))
        # logger.info(f'{points_data}')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_data)[:, :3]) # 只取前三列
        # logger.info(f'{pcd}')
        o3d.io.write_point_cloud(pcd_path, pcd)



def loop_detection_main(args):
    '''
    1. 将bag_file_path解析到文件夹pcd_dir_path
    2. 合并pcd_dir_path到pcd_file_path
    :param bag_file_path:
    :param pcd_dir_path:
    :param pcd_file_path:
    :return:
    '''
    bag_file_path = args.bag_file_path
    pcd_dir_path = args.pcd_dir_path
    pcd_file_path = args.pcd_file_path

    if not os.path.exists(pcd_dir_path):
        os.mkdir(pcd_dir_path)
    try:
        BagExtractor(bag_file_path, pcd_dir_path).run()
    except Exception as e:
        logger.error(f'{bag_file_path} 解析失败, e : {e}')

    # 统计txt文件的数量，如果大于0，则用带里程计矫正的点云拼接，否则用不带里程计矫正的点云拼接
    if len(glob(osp.join(pcd_dir_path, '*.txt'))) > 0:
        # SimpleMerge(pcd_dir_path, pcd_file_path).merge()
        loop_and_merge(args)
    else:
        SimpleMergeWithoutOdometer(pcd_dir_path, pcd_file_path).merge()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('bag_file_path')  # 待解析的bag
    parser.add_argument('pcd_dir_path')   # 临时文件夹
    parser.add_argument('pcd_file_path')  # 解算后的pcd位置
    parser.add_argument('loop_detection', type = int)  # 回环检测


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

    if args.loop_detection == 0:
        main(args.bag_file_path, args.pcd_file_path)
    else:
        loop_detection_main(args )

