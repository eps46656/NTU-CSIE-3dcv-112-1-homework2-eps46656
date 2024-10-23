from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import cv2

from config import *

RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180

images_df = None

def LoadImagesDF():
    global images_df

    if images_df == None:
        images_df = pd.read_pickle(f"{DATA_DIR}/images.pkl")

def SolveP3P_SolveBoA(Cab, Cbc, Cca, Rab, Rbc, Rca):
    K1 = (Rbc / Rca)**2
    K2 = (Rbc / Rab)**2

    G4 =   (K1*K2-K1-K2)**2 \
         - 4*K1*K2*Cbc**2
    G3 =   4*(K1*K2-K1-K2)*K2*(1-K1)*Cab \
         + 4*K1*Cbc*((K1*K2-K1+K2)*Cca+2*K2*Cab*Cbc)
    G2 =   (2*K2*(1-K1)*Cab)**2 \
         + 2*(K1*K2-K1-K2)*(K1*K2+K1-K2) \
         + 4*K1*((K1-K2)*Cbc**2+K1*(1-K1)*Cca**2-2*(1+K1)*K2*Cab*Cbc*Cca)
    G1 =   4*(K1*K2-K1-K2)*K2*(1-K1)*Cab \
         + 4*K1*((K1*K2-K1+K2)*Cbc*Cca+2*K1*K2*Cab*Cca**2)
    G0 =   (K1*K2+K1-K2)**2-4*K1**2*K2*Cca**2

    coeffs = np.array([G4, G3, G2, G1, G0])

    ret = np.roots(coeffs).tolist()

    return ret

def SolveP3P_0_(Cab, Cbc, Cca, Rab, Rbc, Rca):
    ret = []

    for b in [Rab, -Rab]:
        for c in [Rca, -Rca]:
            err = Rbc**2 - Rab**2 - Rbc**2 - 2 * b * c * Cbc

            if err.absolute() <= 1e-4:
                ret.append([0, b, c])

    return ret

def SolveP3P_(Cab, Cbc, Cca, Rab, Rbc, Rca):
    K1 = (Rbc / Rca)**2
    K2 = (Rbc / Rab)**2

    G4 =   (K1*K2-K1-K2)**2 \
        - 4*K1*K2*Cbc**2
    G3 =   4*(K1*K2-K1-K2)*K2*(1-K1)*Cab \
        + 4*K1*Cbc*((K1*K2-K1+K2)*Cca+2*K2*Cab*Cbc)
    G2 =   (2*K2*(1-K1)*Cab)**2 \
        + 2*(K1*K2-K1-K2)*(K1*K2+K1-K2) \
        + 4*K1*((K1-K2)*Cbc**2+K1*(1-K1)*Cca**2-2*(1+K1)*K2*Cab*Cbc*Cca)
    G1 =   4*(K1*K2-K1-K2)*K2*(1-K1)*Cab \
        + 4*K1*((K1*K2-K1+K2)*Cbc*Cca+2*K1*K2*Cab*Cca**2)
    G0 =   (K1*K2+K1-K2)**2-4*K1**2*K2*Cca**2

    xs = np.roots(np.array([G4, G3, G2, G1, G0])).tolist()

    m = 1 - K1
    m_ = 1

    ret = list()

    for x in xs:
        p = 2*(K1*Cca-x*Cbc)
        p_ = -2*x*Cbc
        q = x**2-K1
        q_ = x**2*(1-K2)+2*x*K2*Cab-K2
        y = ((m*q_-m_*q) / (p*m_-p_*m) + (p*q_-p_*q) / (m_*q-m*q_)) / 2
        a = Rab**2 / (1 + x**2 - 2*x*Cab)
        b = x*a
        c = y*a

        ret.append([a, b, c])

    return ret

def SolveP3P(Cab, Cbc, Cca, Rab, Rbc, Rca):
    # a b c
    # a c b

    boa = SolveP3P_SolveBoA(Cab, Cbc, Cca, Rab, Rbc, Rca) # [a, b, c]
    coa = SolveP3P_SolveBoA(Cca, Cbc, Cab, Rca, Rbc, Rab) # [c, a, b]

    aob = SolveP3P_SolveBoA(Cab, Cca, Cbc, Rab, Rca, Rbc) # [b, a, c]
    cob = SolveP3P_SolveBoA(Cbc, Cca, Cab, Rbc, Rca, Rab) # [b, c, a]

    aoc = SolveP3P_SolveBoA(Cbc, Cca, Cab, Rbc, Rca, Rab) # [c, a, a]
    boc = SolveP3P_SolveBoA(Cbc, Cca, Cab, Rbc, Rca, Rab) # [c, b, a]

    pass

    pass


def GetNormalizedMat(points, center, dist):
    # points[P, N]
    # center[P-1]
    # dist

    center = np.array(center)

    assert len(points.shape) == 2
    assert points.shape[0] - 1 == center.shape[0]

    P = points.shape[0]

    origin = points.mean(1)[:-1]

    odist = (((points[:-1, :] - origin.reshape([P-1, 1]))**2).sum(0)**0.5).mean()

    k = dist / odist

    ret = np.zeros([P, P])

    for i in range(P-1):
        ret[i, i] = k

    ret[-1, -1] = 1

    ret[:-1, -1] = center - origin * k

    return ret

def FindHomographic(src, dst):
    # src[P, N]
    # dst[Q, N]

    assert len(src.shape) == 2
    assert len(dst.shape) == 2
    assert src.shape[1] == dst.shape[1]

    P, N = src.shape
    Q, _ = dst.shape

    A = np.zeros([N*(Q-1), P*Q])

    for i in range(N):
        for j in range(Q-1):
            A[(Q-1)*i+j, j*P:j*P+P] = src[:, i]
            A[(Q-1)*i+j, -P:] = src[:, i] * -dst[j, i]

    _, _, Vh = np.linalg.svd(A)

    return Vh[-1, :].reshape([Q, P])

def HomographyTrans(H, x):
    assert len(H.shape) == 2
    assert len(x.shape) == 2

    ret = H @ x

    P = ret.shape[0]

    for i in range(P):
        np.divide(ret[i, :], ret[-1, :], out=ret[i, :])

    return ret

def NormalizedDLT(points1, points2, normalized):
    # points1[P, N]
    # points2[Q, N]

    assert len(points1.shape) == 2
    assert len(points2.shape) == 2

    P = points1.shape[0]
    Q = points2.shape[0]

    T1 = np.identity(P)
    T2 = np.identity(Q)

    if normalized:
        T1 = GetNormalizedMat(points1, np.zeros([P-1]), np.sqrt(P-1))
        T2 = GetNormalizedMat(points2, np.zeros([Q-1]), np.sqrt(Q-1))

    rep_points1 = T1 @ points1
    rep_points2 = T2 @ points2

    H = FindHomographic(rep_points1, rep_points2) # [P, Q]

    return T1, T2, H
    # return np.linalg.inv(T2) @ H @ T1

def CalcErr(points1, points2):
    # points1[P, N]
    # points2[P, N]

    assert len(points1.shape) == 2
    assert len(points2.shape) == 2
    assert points1.shape == points2.shape

    return np.sqrt(((points1 - points2)**2).sum(0))

def FindNearestRotMat(M):
    assert M.shape == (3, 3)
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh

def KRTDecompose(H):
    assert H.shape == (3, 4)

    def Normalize(x):
        norm = np.linalg.norm(x)
        return norm, x / norm

    H = H / np.linalg.norm(H[2, :3])

    if np.linalg.det(H[:, :3]):
        H *= -1

    p0 = H[0, :3]
    p1 = H[1, :3]
    p2 = H[2, :3]

    r2 = p2
    oy = np.dot(p1, r2)
    fy, r1 = Normalize(p1 - oy * r2)
    os = np.dot(p0, r2)
    s = np.dot(p0, r1)
    fx, r0 = Normalize(p0 - os * r2 - s * r1)

    t2 = H[2, 3]
    t1 = (H[1, 3] - oy * t2) / fy
    t0 = (H[0, 3] - os * t2 - s * t1) / fx

    K = np.array([
        [fx,  s, os],
        [ 0, fy, oy],
        [ 0,  0,  1],])

    R = np.array([r0, r1, r2])

    T = np.array([[t0], [t1], [t2]])

    return K, R, T

def SolvePnP(points_3d, points_2d, camera_mat, dist_coeffs):
    # points_3d[N, 3]
    # points_2d[N, 2]

    N = points_3d.shape[0]

    assert points_3d.shape == (N, 3)
    assert points_2d.shape == (N, 2)
    assert points_3d.dtype == points_2d.dtype

    points_2d = cv2.undistortPoints(points_2d,
                                    camera_mat,
                                    dist_coeffs,
                                    None,
                                    camera_mat).reshape([N, 2])

    ones = np.ones([1, N], dtype=points_3d.dtype)
    points_3d = np.concatenate([points_3d.transpose(), ones], 0) # [4, N]
    points_2d = np.concatenate([points_2d.transpose(), ones], 0) # [3, N]

    T1, T2, H = NormalizedDLT(points_3d, points_2d, True)

    H = np.linalg.inv(T2) @ H @ T1

    rt = np.linalg.inv(camera_mat) @ H

    r = rt[:, :3]
    t = rt[:, 3]

    k = 1 / np.linalg.det(r)
    k = -(-k)**(1/3) if k < 0 else k**(1/3)

    r *= k
    t *= k

    r = FindNearestRotMat(r)

    return True, R.from_matrix(r).as_rotvec(), t, []

def GetIdxToId():
    global images_df

    LoadImagesDF()

    ps = []

    for id in range(1, 293+1):
        name = ((images_df.loc[images_df["IMAGE_ID"] == id])["NAME"].values)[0]

        if name.startswith("valid"):
            continue

        frame_k = int(name.strip("train_img").strip(".jpg"))
        ps.append((frame_k, id))

    ps.sort(key=lambda t: t[0])
    ret = np.array([p[1] for p in ps])

    return ret

def ReadRT(filename):
    rvecs = None
    tvecs = None

    with open(filename, "rb") as f:
        rvecs = np.load(f)
        tvecs = np.load(f)

    return rvecs, tvecs
