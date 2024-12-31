
import copy

import open3d as o3d
import numpy as np

import os
import os.path as osp
import time
import yaml
from scipy.spatial.transform import Rotation as R


vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)

vis_pcd = o3d.geometry.PointCloud()
vis.add_geometry(vis_pcd)

PCDS_DIR = r"C:\Users\jkkc\Desktop\TEST\1_transed"
file_list = [f for f in os.listdir(PCDS_DIR) if f.endswith('.pcd')]
to_reset = True

coordinate_sys = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)


def get_pose_from(txt_full_path):
    '''
    辅助函数：从txt文件中读取平移向量xyz和旋转四元数
    返回平移向量和旋转矩阵
    '''
    with open(txt_full_path, 'r', encoding='utf-8') as f:
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
    # 将四元数转为欧拉角
    euler_angle = R.from_quat(quaternion).as_euler('xyz', degrees=True)

    # 将四元数转换为旋转矩阵
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    # 返回pos, rotate, roll
    return translation_vector, rotation_matrix, euler_angle[0]



for i, f in enumerate(file_list):

    pcd_filename = osp.join(PCDS_DIR, f)
    pcd:o3d.geometry.PointCloud = o3d.io.read_point_cloud(pcd_filename)
    vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))

    # vis.update_geometry()
    # 注意，如果使用的是open3d 0.8.0以后的版本，这句话应该改为下面格式
    vis.update_geometry(vis_pcd)

    if to_reset:
        vis.reset_view_point(True)
        to_reset = False

    vis.poll_events()
    vis.update_renderer()
    # if f == '1703038968.250.pcd' or f == '1703039146.152.pcd':
    if i == 1830 or i == 1709:
        time.sleep(2)



vis.destroy_window()



