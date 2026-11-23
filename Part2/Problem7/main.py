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

def distance_to_circle(points):
    """Shortest distance to the circle, independent of traversal speed."""
    radial = np.hypot(points[:, 1], points[:, 2] - 120) - 32
    return np.hypot(points[:, 0] - 150, radial)


# Radial deviation from the desired circle
radial_error = np.abs(
    np.sqrt(actual_points[:, 1]**2 +
            (actual_points[:, 2] - 120)**2) - 32
)

# Deviation from the desired X plane
x_error = np.abs(actual_points[:, 0] - 150)

# Total distance to the desired circle
path_error = distance_to_circle(actual_points)

if __name__ == "__main__":
    print("Maximum radial error (mm):", np.max(radial_error))
    print("Maximum X deviation (mm):", np.max(x_error))
    print("Maximum path error (mm):", np.max(path_error))
    from Part2.Problem7.improve import improved_points_9, improved_points_37

    print("\nDistance to the desired circle (sampled every 0.02 s):")
    print("Knots   Maximum (mm)   Mean (mm)")
    for n_knots, points in [(5, actual_points), (9, improved_points_9),
                            (37, improved_points_37)]:
        error = distance_to_circle(points)
        print(f"{n_knots:5d}   {error.max():12.6f}   {error.mean():9.6f}")
