import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp
from Part1.Problem1.main import T04, q1, q2, q3, q4
from Part1.Problem2.main import inverse_kinematics


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


# Forward-kinematics results are also used by visualize.py.
fk = sp.lambdify((q1, q2, q3, q4), T04, "numpy")
transforms_fk = np.array([fk(*qi) for qi in q])
positions_fk = transforms_fk[:, :3, 3]
orientation_z = transforms_fk[:, 2, 0]
position_errors = np.linalg.norm(points - positions_fk, axis=1)

if __name__ == "__main__":
    print("Elbow-up configurations (angles in rad):")
    print(" j  phi (deg)       q1        q2        q3        q4")
    for j, angles in enumerate(q):
        print(f"{j:2d}  {j*10:9.0f} " + " ".join(f"{angle:9.4f}" for angle in angles))
    print("Maximum position error (mm):", np.max(position_errors))
    print("Maximum orientation error:", np.max(np.abs(orientation_z)))
