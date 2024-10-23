from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np

from config import *

images_df = pd.read_pickle(f"{DATA_DIR}/images.pkl")

rvec_gts = np.zeros([294, 3])
tvec_gts = np.zeros([294, 3])

for id in range(1, 293+1):
    # Get camera pose groudtruth
    ground_truth = images_df.loc[images_df["IMAGE_ID"] == id]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values.reshape([4])
    tvec_gt = ground_truth[["TX","TY","TZ"]].values.reshape([3])
    rvec_gt = R.from_quat(rotq_gt).as_rotvec()

    rvec_gts[id] = rvec_gt
    tvec_gts[id] = tvec_gt

with open("rt_gt.np", "wb") as f:
    np.save(f, rvec_gts)
    np.save(f, tvec_gts)
