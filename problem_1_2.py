from scipy.spatial.transform import Rotation as R
import numpy as np

from config import *
from utils import *

rvec_gts, tvec_gts = ReadRT("rt_gt.np")
rvecs, tvecs = ReadRT("rt.np")

r_errs = list()
t_errs = list()

for id in range(1, 294):
    rotm = R.from_rotvec(rvecs[id]).as_matrix()
    rotm_gt = R.from_rotvec(rvec_gts[id]).as_matrix()

    rote = rotm @ np.linalg.inv(rotm_gt)

    r_err = np.linalg.norm(R.from_matrix(rote).as_rotvec())
    t_err = np.linalg.norm(tvecs[id] - tvec_gts[id])

    r_errs.append(r_err)
    t_errs.append(t_err)

r_errs.sort()
t_errs.sort()

print(f"median of r_err = {r_errs[146]} rad")
print(f"median of t_err = {t_errs[146]}")
