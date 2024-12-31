import open3d as o3d
import numpy as np
import copy
import yaml
from scipy.spatial.transform import Rotation as R


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

loop_frame_path = 'C:/Users/jkkc/Desktop/TEST/1/1703038954.652.pcd'
loop_odo_path = 'C:/Users/jkkc/Desktop/TEST/1/1703038954.652.txt'
curr_frame_path = 'C:/Users/jkkc/Desktop/TEST/1/1703039136.148.pcd'
curr_odo_path = 'C:/Users/jkkc/Desktop/TEST/1/1703039136.148.txt'

relative_transformation = np.array([[ 9.9900e-01, -1.6743e-04, -4.4792e-02,  5.2065e-01],
 [ 1.6117e-02,  9.3436e-01,  3.5597e-01, 1.0754e+00],
 [ 4.1792e-02, -3.5634e-01,  9.3342e-01,  1.9941e-01],
 [ 0.0000e+00,  0.0000e+00, 0.0000e+00,  1.0000e+00],
 ])

loop_frame2:o3d.geometry.PointCloud = o3d.io.read_point_cloud(loop_frame_path)
loop_pose = get_transformation(loop_odo_path)
curr_frame2 = o3d.io.read_point_cloud(curr_frame_path)
curr_pose = get_transformation(curr_odo_path)

loop_frame1 = copy.deepcopy(loop_frame2)
curr_frame1 = copy.deepcopy(curr_frame2)

loop_frame1.transform(loop_pose)
curr_frame1.transform(curr_pose)

loop_frame1 = loop_frame1.translate((10,10,10))
curr_frame1 = curr_frame1.translate((10,10,10))

loop_frame1.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(loop_frame1.points), 1)))
curr_frame1.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(curr_frame1.points), 1)))



curr_frame_transed = copy.deepcopy(curr_frame2)
curr_frame_transed.transform(relative_transformation)

loop_frame2.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(loop_frame2.points), 1)))
curr_frame2.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(curr_frame2.points), 1)))
curr_frame_transed.colors = o3d.utility.Vector3dVector(np.tile([0,0,1], (len(curr_frame_transed.points), 1)))

o3d.visualization.draw_geometries([loop_frame1, curr_frame1, loop_frame2, curr_frame2, curr_frame_transed])
# o3d.visualization.draw_geometries([loop_frame, curr_frame, loop_frame_transed])