import numpy as np
import sympy as sp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Part1.Problem2.main import o04, x04
from Part1.Problem1.main import T04, q1 as s1, q2 as s2, q3 as s3, q4 as s4

R = 32
pc = np.array([150, 0, 120])

phi = np.linspace(0, 2*np.pi, 37)

points = np.array([
    pc[0] + np.zeros_like(phi),
    pc[1] + R * np.cos(phi),
    pc[2] + R * np.sin(phi)
]).T


# Joint 1 angles
q1 = np.arctan2(points[:, 1], points[:, 0])

# Wrist coordinates in the arm plane
rw = np.sqrt(points[:, 0]**2 + points[:, 1]**2) - 50
zw = points[:, 2] - 50

L2 = L3 = 93

cos_q3 = (rw**2 + zw**2 - L2**2 - L3**2) / (2*L2*L3)
q3 = np.arccos(np.clip(cos_q3, -1, 1))

alpha = np.arctan2(zw, rw)

beta = np.arctan2(
    L3 * np.sin(q3),
    L2 + L3 * np.cos(q3)
)

q2 = alpha - beta

q4 = -q2 - q3

q = np.column_stack((q1, q2, q3, q4))


# Results validation ==========================
# Convert symbolic FK into a numerical function
fk = sp.lambdify((s1, s2, s3, s4), T04, "numpy")

positions_fk = []
orientation_z = []

for qi in q:
    T = fk(*qi)

    positions_fk.append(T[:3, 3])
    orientation_z.append(T[2, 0])

positions_fk = np.array(positions_fk)

# Compare desired and actual positions
position_errors = np.linalg.norm(points - positions_fk, axis=1)
# =============================================

if __name__ == "__main__":
    print(points.shape)
    print(f"Point 0: {points[0][0]:.2f}, {points[0][1]:.2f}, {points[0][2]:.2f}")
    print(f"Point 9: {points[9][0]:.2f}, {points[9][1]:.2f}, {points[9][2]:.2f}")
    print(f"Point 18: {points[18][0]:.2f}, {points[18][1]:.2f}, {points[18][2]:.2f}")
    print(f"Point 36: {points[36][0]:.2f}, {points[36][1]:.2f}, {points[36][2]:.2f}")

    theta = np.degrees(np.arctan2(32, 150))
    print(f"1st point angle: {theta:.2f} degrees")

    print("Shape:", q.shape)
    print("First configuration (degrees):", np.degrees(q[0]))
    print("Last configuration (degrees):", np.degrees(q[-1]))

    # Validate FK results
    print("Maximum position error (mm):", np.max(position_errors))
    print("Maximum orientation error:", np.max(np.abs(orientation_z)))