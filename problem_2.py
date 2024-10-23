import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

from config import *
from utils import *

def GetCameraRT(frame_idx):
    global rotqs, tvecs

    rotq = rotqs[frame_idx]
    rotm = R.from_quat(rotq.reshape([4])).as_matrix() # [3, 3]
    tvec = tvecs[frame_idx] # [3]

    ret = np.empty([3, 4])
    ret[:, :3] = rotm
    ret[:, 3] = tvec

    return ret

def GetMGrid(N):
    inc = 1 / (N + 1)
    x = np.mgrid[:N, :N].reshape([2, N*N])
    x = x * (1 - 2 * inc) / (N - 1) + inc
    return x

def GetPointCube(density, transform):
    assert transform.shape == (4, 4)

    ones = np.ones([density**2])
    zeros = np.zeros([density**2])
    mgrid = GetMGrid(density) # [2, density**2]

    vertices = [
        np.stack([   zeros, mgrid[0], mgrid[1], ones]),  # nx
        np.stack([mgrid[0],    zeros, mgrid[1], ones]),  # ny
        np.stack([mgrid[0], mgrid[1],    zeros, ones]),  # nz
        np.stack([    ones, mgrid[0], mgrid[1], ones]),  # px
        np.stack([mgrid[0],     ones, mgrid[1], ones]),  # py
        np.stack([mgrid[0], mgrid[1],     ones, ones]),] # pz
    # [4, density**2]

    colors = [
        np.array([255,   0,   0], dtype=np.uint8),  # nx
        np.array([  0, 255, 255], dtype=np.uint8),  # ny
        np.array([  0, 255,   0], dtype=np.uint8),  # nz
        np.array([255,   0, 255], dtype=np.uint8),  # px
        np.array([  0,   0, 255], dtype=np.uint8),  # py
        np.array([255, 255,   0], dtype=np.uint8),] # pz

    for k in range(6):
        vertices[k] = (transform @ vertices[k]).transpose()
            # [density**2, 4]
        colors[k] = colors[k].reshape([1, 3]).repeat(density**2, axis=0)

    # vertices[i] #[density**2, 4]
    # colors[i] #[density**2, 3]

    ret_vertices = np.stack(vertices).reshape([6 * density**2, 4])
    ret_colors = np.stack(colors).reshape([6 * density**2, 3])

    return ret_vertices, ret_colors

def DrawPoints(img, points, colors):
    N = points.shape[0]

    assert points.shape == (N, 3)
    assert colors.shape == (N, 3)
    assert len(img.shape) == 3

    H, W, _ = img.shape

    idxes = np.argsort(-points[:, 2]) # [N]

    points = points[idxes, :] # [N, 3]
    colors = colors[idxes, :] # [N, 3]

    radius = 3
    thickness = radius * 2

    for i in range(N):
        d = points[i][2]
        w = int(np.round(points[i][0] / d))
        h = int(np.round(points[i][1] / d))

        if h < 0 or H <= h or w < 0 or W <= w:
            continue

        color = tuple(int(c) for c in colors[i])

        cv2.circle(img, (w, h), radius, color, thickness)

def GetFrame(id):
    global cube_density, cube_transform

    frame_name = ((images_df.loc[images_df["IMAGE_ID"] == id])["NAME"].values)[0]
    frame = cv2.imread(f"{DATA_DIR}/frames/{frame_name}")

    vertices, colors = GetPointCube(cube_density, cube_transform)
    # vertices[N, 4]
    # colors[N, 3]

    N = vertices.shape[0]

    RT = np.empty([3, 4])
    RT[:, :3] = R.from_rotvec(rvecs[id]).as_matrix()
    RT[:, 3] = tvecs[id]

    points = vertices @ (camera_matrix @ RT).transpose() # [N, 3]

    DrawPoints(frame, points, colors)

    return frame

images_df = pd.read_pickle(f"{DATA_DIR}/images.pkl")

camera_matrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])

rvecs, tvecs = ReadRT("rt.np")

cube_density = 24

cube_transform = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],])

idx_to_id = GetIdxToId()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vout = cv2.VideoWriter("problem_2_2.mp4", fourcc, 8, [1080, 1920], True)

for idx in range(idx_to_id.shape[0]):
    print(f"idx = {idx}")
    frame = GetFrame(idx_to_id[idx])
    vout.write(frame)

vout.release()
