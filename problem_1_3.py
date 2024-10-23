import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd

from config import *
from utils import *

def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()

    axes.points = o3d.utility.Vector3dVector([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    axes.lines  = o3d.utility.Vector2iVector([
        [0, 1], [0, 2], [0, 3]]) # X, Y, Z

    axes.colors = o3d.utility.Vector3dVector([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B

    return axes

def load_camera_trajectory():
    camera_trajectory = o3d.geometry.LineSet()

    camera_trajectory.points = o3d.utility.Vector3dVector([
        camera_Ts[idx_to_id[idx]] for idx in range(idx_to_id.shape[0])])

    camera_trajectory.lines = o3d.utility.Vector2iVector([
        [idx, idx-1] for idx in range(1, idx_to_id.shape[0])])

    camera_trajectory.colors = o3d.utility.Vector3dVector([
        [0, 1, 0] for idx in range(1, idx_to_id.shape[0])])

    return camera_trajectory

def load_camera_cone():
    camera_cone = o3d.geometry.LineSet()

    camera_cone.points = o3d.utility.Vector3dVector([
        camera_origin,
        camera_tl,
        camera_tr,
        camera_br,
        camera_bl,])

    camera_cone.lines = o3d.utility.Vector2iVector([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],])

    camera_cone.colors = o3d.utility.Vector3dVector([
        [1, 0, 0] for i in range(8)])

    return camera_cone

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

def UpdateCameraCone(id):
    global vis, camera_cone, camera_Rs, camera_Ts

    camera_R = camera_Rs[id].transpose()
    camera_T = camera_Ts[id]

    camera_cone.points = o3d.utility.Vector3dVector([
        camera_origin + camera_T,
        camera_tl @ camera_R + camera_T,
        camera_tr @ camera_R + camera_T,
        camera_br @ camera_R + camera_T,
        camera_bl @ camera_R + camera_T,])

    vis.update_geometry(camera_cone)

def PrevCamera(vis):
    global cur_frame_idx

    if 0 <= cur_frame_idx - 1:
        cur_frame_idx -= 1

    UpdateCameraCone(idx_to_id[cur_frame_idx])

def NextCamera(vis):
    global cur_frame_idx

    if cur_frame_idx + 1 < idx_to_id.shape[0]:
        cur_frame_idx += 1

    UpdateCameraCone(idx_to_id[cur_frame_idx])

H, W = 1920, 1080

camera_matrix = np.array([[1868.27,   0   ,540],
                          [   0   ,1869.18,960],
                          [   0   ,   0   ,  1]])

inv_camera_matrix = np.linalg.inv(camera_matrix)

camera_origin = np.array([0, 0, 0])
camera_tl = (inv_camera_matrix @ np.array([[0], [0], [1]])).reshape([3])
camera_tr = (inv_camera_matrix @ np.array([[W], [0], [1]])).reshape([3])
camera_br = (inv_camera_matrix @ np.array([[W], [H], [1]])).reshape([3])
camera_bl = (inv_camera_matrix @ np.array([[0], [H], [1]])).reshape([3])

camera_rvecs, camera_tvecs = ReadRT("rt.np")

camera_Rs = [None for i in range(294)]
camera_Ts = [None for i in range(294)]

for id in range(1, 294):
    camera_Rs[id] = np.linalg.inv(
        R.from_rotvec(camera_rvecs[id]).as_matrix())
    camera_Ts[id] = -camera_tvecs[id] @ camera_Rs[id].transpose()

idx_to_id = GetIdxToId()

cur_frame_idx = 0

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# load point cloud
points3D_df = pd.read_pickle(f"{DATA_DIR}/points3D.pkl")
pcd = load_point_cloud(points3D_df)
vis.add_geometry(pcd)

# load axes
axes = load_axes()
vis.add_geometry(axes)

camera_trajectory = load_camera_trajectory()
vis.add_geometry(camera_trajectory)

# load camera cone
camera_cone = load_camera_cone()
vis.add_geometry(camera_cone)
UpdateCameraCone(idx_to_id[0])

R_euler = np.array([0, 0, 0]).astype(float)
t = np.array([0, 0, 0]).astype(float)
scale = 1.0

# just set a proper initial camera view
vc = vis.get_view_control()
vc_cam = vc.convert_to_pinhole_camera_parameters()
initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
initial_cam[-1, -1] = 1.
setattr(vc_cam, 'extrinsic', initial_cam)
vc.convert_from_pinhole_camera_parameters(vc_cam)

# set key callback
shift_pressed = False
vis.register_key_callback(ord('A'), PrevCamera)
vis.register_key_callback(ord('D'), NextCamera)

print('[Keyboard usage]')
print('Prev Camera Cone\tA')
print('Next Camera Cone\tD')

vis.run()
vis.destroy_window()
