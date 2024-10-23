from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2

from config import *
import utils

images_df = pd.read_pickle(f"{DATA_DIR}/images.pkl")
train_df = pd.read_pickle(f"{DATA_DIR}/train.pkl")
points3D_df = pd.read_pickle(f"{DATA_DIR}/points3D.pkl")
point_desc_df = pd.read_pickle(f"{DATA_DIR}/point_desc.pkl")

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return utils.SolvePnP(points3D, points2D, cameraMatrix, distCoeffs)

    # return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)

# Process model descriptors
desc_df = average_desc(train_df, points3D_df)
kp_model = np.array(desc_df["XYZ"].to_list())
desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

print("start")

rvecs = np.zeros([294, 3])
tvecs = np.zeros([294, 3])

for id in range(1, 294):
    print(f"id = {id}")

    # Load quaery image
    fname = ((images_df.loc[images_df["IMAGE_ID"] == id])["NAME"].values)[0]
    rimg = cv2.imread(f"{DATA_DIR}/frames/{fname}", cv2.IMREAD_GRAYSCALE)

    # Load query keypoints and descriptors
    points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == id]
    kp_query = np.array(points["XY"].to_list())
    desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

    # Find correspondance and solve pnp
    retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
    rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()
    tvec = tvec.reshape(1,3)
    rotm = R.from_quat(rotq.reshape([4])).as_matrix()

    rvecs[id] = rvec.reshape([3])
    tvecs[id] = tvec.reshape([3])

    # Get camera pose groudtruth
    ground_truth = images_df.loc[images_df["IMAGE_ID"] == id]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values
    rotm_gt = R.from_quat(rotq_gt.reshape([4])).as_matrix()

    rote = rotm @ np.linalg.inv(rotm_gt)

    err_ang = np.linalg.norm(R.from_matrix(rote).as_rotvec())
    err_dist = np.linalg.norm(tvec - tvec_gt)

    print(f"err_ang = {err_ang}")
    print(f"err_dist = {err_dist}")

with open("rt.np", "wb") as f:
    np.save(f, rvecs)
    np.save(f, tvecs)

print("\a")