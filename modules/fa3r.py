import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Vector3d:
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

@dataclass
class Matrix3d:
    data: np.ndarray

    def __post_init__(self):
        if self.data.shape != (3, 3):
            raise ValueError("Matrix3d must be 3x3")

shift = 20
d2l = 2.0 ** shift
shifted1 = 2 << shift
shifted2 = shift << 1
shifted3 = 2 << (3 * shift + 1)
l2d = 1.0 / d2l

def cross_int(x: Tuple[int, int, int], y: Tuple[int, int, int], k: int, z: List[int]) -> None:
    z[0] = (k * (z[0] + ((x[1] * y[2] - x[2] * y[1]) >> shift))) >> shifted2
    z[1] = (k * (z[1] + ((x[2] * y[0] - x[0] * y[2]) >> shift))) >> shifted2
    z[2] = (k * (z[2] + ((x[0] * y[1] - x[1] * y[0]) >> shift))) >> shifted2

def cross_double(x: np.ndarray, y: np.ndarray, k: float, z: np.ndarray) -> None:
    z[0] = k * (z[0] + x[1] * y[2] - x[2] * y[1])
    z[1] = k * (z[1] + x[2] * y[0] - x[0] * y[2])
    z[2] = k * (z[2] + x[0] * y[1] - x[1] * y[0])

def compute_sigma(P: List[Vector3d], Q: List[Vector3d]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(P)
    mean_X = np.mean([p.to_numpy() for p in P], axis=0)
    mean_Y = np.mean([q.to_numpy() for q in Q], axis=0)
    sigma = np.zeros((3, 3))
    for i in range(n):
        sigma += np.outer(Q[i].to_numpy() - mean_Y, P[i].to_numpy() - mean_X)
    sigma /= n
    return sigma, mean_X, mean_Y

def FA3R_int(P: Optional[List[Vector3d]] = None,
             Q: Optional[List[Vector3d]] = None,
             sigma: Optional[np.ndarray] = None,
             num: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if P is not None and Q is not None and sigma is None:
        sigma, mean_X, mean_Y = compute_sigma(P, Q)
    else:
        mean_X = np.zeros(3)
        mean_Y = np.zeros(3)

    max_val = np.max(np.abs(sigma))
    h = np.int64(sigma / max_val * d2l)

    for _ in range(num):
        h_ = h.copy()
        k = shifted3 // (np.sum(h_**2) // shift + shifted1)
        
        cross_int(h_[0], h_[1], k, h[2])
        cross_int(h_[2], h_[0], k, h[1])
        cross_int(h_[1], h_[2], k, h[0])

    R = (h * l2d).T
    R = R / np.linalg.norm(R, axis=0)
    t = mean_X - R.T @ mean_Y

    return R, t

def FA3R_double(P: Optional[List[Vector3d]] = None,
                Q: Optional[List[Vector3d]] = None,
                sigma: Optional[np.ndarray] = None,
                num: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if P is not None and Q is not None and sigma is None:
        sigma, mean_X, mean_Y = compute_sigma(P, Q)
    else:
        mean_X = np.zeros(3)
        mean_Y = np.zeros(3)

    max_val = np.max(np.abs(sigma))
    h = sigma / max_val

    for _ in range(num):
        h_ = h.copy()
        k = 2.0 / (np.sum(h_**2) + 1.0)
        
        cross_double(h_[0], h_[1], k, h[2])
        cross_double(h_[2], h_[0], k, h[1])
        cross_double(h_[1], h_[2], k, h[0])

    R = h.T
    R = R / np.linalg.norm(R, axis=0)
    t = mean_X - R.T @ mean_Y

    return R, t

def eig3D_eig(P: Optional[List[Vector3d]] = None,
              Q: Optional[List[Vector3d]] = None,
              sigma: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if P is not None and Q is not None and sigma is None:
        sigma, mean_X, mean_Y = compute_sigma(P, Q)
    else:
        mean_X = np.zeros(3)
        mean_Y = np.zeros(3)

    A = sigma - sigma.T
    D = np.array([A[1, 2], A[2, 0], A[0, 1]])
    QQ = np.zeros((4, 4))
    QQ[0, 0] = np.trace(sigma)
    tmp = sigma + sigma.T
    np.fill_diagonal(tmp, tmp.diagonal() - QQ[0, 0])
    QQ[0, 1:] = D
    QQ[1:, 0] = D
    QQ[1:, 1:] = tmp

    eigenvalues, eigenvectors = np.linalg.eig(QQ)
    max_index = np.argmax(eigenvalues)
    q = eigenvectors[:, max_index]
    q = q / np.linalg.norm(q)

    R = np.array([
        [-q[2]**2-q[3]**2, q[1]*q[2]-q[0]*q[3], q[1]*q[3]+q[0]*q[2]],
        [q[1]*q[2]+q[0]*q[3], -q[1]**2-q[3]**2, q[2]*q[3]-q[0]*q[1]],
        [q[1]*q[3]-q[0]*q[2], q[2]*q[3]+q[0]*q[1], -q[1]**2-q[2]**2]
    ]) * 2 + np.eye(3)

    t = mean_X - R.T @ mean_Y

    return R, t

def eig3D_symbolic(P: Optional[List[Vector3d]] = None,
                   Q: Optional[List[Vector3d]] = None,
                   sigma: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if P is not None and Q is not None and sigma is None:
        sigma, mean_X, mean_Y = compute_sigma(P, Q)
    else:
        mean_X = np.zeros(3)
        mean_Y = np.zeros(3)

    A = sigma - sigma.T
    D = np.array([A[1, 2], A[2, 0], A[0, 1]])
    QQ = np.zeros((4, 4))
    QQ[0, 0] = np.trace(sigma)
    tmp = sigma + sigma.T
    np.fill_diagonal(tmp, tmp.diagonal() - QQ[0, 0])
    QQ[0, 1:] = D
    QQ[1:, 0] = D
    QQ[1:, 1:] = tmp

    c = np.linalg.det(QQ)
    b = -8.0 * np.linalg.det(sigma)
    a = -2.0 * np.sum(sigma**2)

    T0 = 2.0 * a**3 + 27.0 * b**2 - 72.0 * a * c
    tt = a**2 + 12.0 * c
    theta = np.arctan2(np.sqrt(4.0 * tt**3 - T0**2), T0)
    aT1 = 1.259921049894873 * np.sqrt(tt) * np.cos(theta / 3)
    T2 = np.sqrt(-4.0 * a + 3.174802103936399 * aT1)
    lambda_ = 0.204124145231932 * (T2 + np.sqrt(-T2**2 - 12.0 * a - 29.393876913398135 * b / T2))

    G = QQ.copy()
    G[0, 0] -= lambda_
    G[1, 1] -= lambda_
    G[2, 2] -= lambda_
    G[3, 3] -= lambda_

    q = np.array([
        G[0, 3]*G[1, 2]*G[1, 2] - G[0, 2]*G[1, 2]*G[1, 3] - G[0, 3]*G[1, 1]*G[2, 2] + G[0, 1]*G[1, 3]*G[2, 2] + G[0, 2]*G[1, 1]*G[2, 3] - G[0, 1]*G[1, 2]*G[2, 3],
        G[0, 2]*G[0, 2]*G[1, 3] + G[0, 1]*G[0, 3]*G[2, 2] - G[0, 0]*G[1, 3]*G[2, 2] + G[0, 0]*G[1, 2]*G[2, 3] - G[0, 2]*G[0, 3]*G[1, 2] - G[0, 2]*G[0, 1]*G[2, 3],
        G[0, 2]*G[0, 3]*G[1, 1] - G[0, 1]*G[0, 3]*G[1, 2] - G[0, 1]*G[0, 2]*G[1, 3] + G[0, 0]*G[1, 2]*G[1, 3] + G[0, 1]*G[0, 1]*G[2, 3] - G[0, 0]*G[1, 1]*G[2, 3],
        -(G[0, 2]*G[0, 2]*G[1, 1] - 2*G[0, 1]*G[0, 2]*G[1, 2] + G[0, 0]*G[1, 2]*G[1, 2] + G[0, 1]*G[0, 1]*G[2, 2] - G[0, 0]*G[1, 1]*G[2, 2])
    ])
    q = q / np.linalg.norm(q)

    R = np.array([
        [1-2*(q[1]**2+q[2]**2), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
        [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]**2+q[2]**2), 2*(q[1]*q[2]-q[0]*q[3])],
        [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]**2+q[1]**2)]
    ])

    t = mean_X - R.T @ mean_Y

    return R, t
