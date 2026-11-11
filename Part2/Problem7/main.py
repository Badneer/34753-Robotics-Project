import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp

from Part1.Problem1.main import T04, q1, q2, q3, q4
from Part2.Problem6.main import trajectory

# Numerical forward kinematics
fk = sp.lambdify((q1, q2, q3, q4), T04, "numpy")

# Evaluate the 8-second trajectory
times = np.linspace(0, 8, 401)

actual_points = []

for t in times:
    q_t, _, _ = trajectory(t)
    T = fk(*q_t)
    actual_points.append(T[:3, 3])

actual_points = np.array(actual_points)

# Radial deviation from the desired circle
radial_error = np.abs(
    np.sqrt(actual_points[:, 1]**2 +
            (actual_points[:, 2] - 120)**2) - 32
)

# Deviation from the desired X plane
x_error = np.abs(actual_points[:, 0] - 150)

# Total distance to the desired circle
path_error = np.sqrt(radial_error**2 + x_error**2)

if __name__ == "__main__":
    print("Maximum radial error (mm):", np.max(radial_error))
    print("Maximum X deviation (mm):", np.max(x_error))
    print("Maximum path error (mm):", np.max(path_error))