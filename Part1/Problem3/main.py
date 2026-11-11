import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp
from Part1.Problem1.main import T04, q1, q2, q3, q4

# Inverse kinematics for horizontal stylus
def inverse_kinematics(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Base angle
    q1_values = np.arctan2(y, x)

    # Wrist position
    rw = np.sqrt(x**2 + y**2) - 50
    zw = z - 50

    # Elbow angle
    L2 = L3 = 93
    cos_q3 = (rw**2 + zw**2 - L2**2 - L3**2) / (2*L2*L3)

    if np.any(np.abs(cos_q3) > 1 + 1e-10):
        raise ValueError("Unreachable target position")

    q3_values = np.arccos(np.clip(cos_q3, -1, 1))

    # Shoulder angle
    alpha = np.arctan2(zw, rw)
    beta = np.arctan2(
        L3 * np.sin(q3_values),
        L2 + L3 * np.cos(q3_values)
    )

    q2_values = alpha - beta

    # Horizontal stylus
    q4_values = -q2_values - q3_values

    return np.column_stack((
        q1_values, q2_values, q3_values, q4_values
    ))


# Problem 3: Circular path
R = 32
pc = np.array([150, 0, 120])

phi = np.linspace(0, 2*np.pi, 37)

points = np.column_stack((
    np.full(37, pc[0]),
    pc[1] + R * np.cos(phi),
    pc[2] + R * np.sin(phi)
))

# 37 robot configurations
q = inverse_kinematics(points)


# Validation using forward kinematics
if __name__ == "__main__":

    fk = sp.lambdify((q1, q2, q3, q4), T04, "numpy")

    positions_fk = []
    orientation_z = []

    for qi in q:
        T = fk(*qi)
        positions_fk.append(T[:3, 3])
        orientation_z.append(T[2, 0])

    positions_fk = np.array(positions_fk)

    position_errors = np.linalg.norm(
        points - positions_fk, axis=1
    )

    print("Configurations shape:", q.shape)
    print("First configuration (degrees):", np.degrees(q[0]))
    print("Last configuration (degrees):", np.degrees(q[-1]))
    print("Maximum position error (mm):", np.max(position_errors))
    print("Maximum orientation error:", np.max(np.abs(orientation_z)))
