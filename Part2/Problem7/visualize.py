
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from Part2.Problem7.main import actual_points, distance_to_circle, times
from Part2.Problem7.improve import improved_points_9, improved_points_37

# Desired circular path
phi = np.linspace(0, 2*np.pi, 401)

y_desired = 32 * np.cos(phi)
z_desired = 120 + 32 * np.sin(phi)

plt.figure(figsize=(7, 7))

# Desired circle
plt.plot(y_desired, z_desired, 'k--',
         label="Desired circle")

# Original trajectory (5 knots)
plt.plot(actual_points[:, 1], actual_points[:, 2],
         label="5 knots")

# Improved trajectory (9 knots)
plt.plot(improved_points_9[:, 1], improved_points_9[:, 2],
         label="9 knots")
plt.plot(improved_points_37[:, 1], improved_points_37[:, 2],
         label="37 knots")

plt.xlabel("Y (mm)")
plt.ylabel("Z (mm)")
plt.title("Problem 7 - Cartesian Path Comparison")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Shortest distance to the desired circle throughout the motion.
plt.figure(figsize=(9, 4))
for label, points in [("5 knots", actual_points), ("9 knots", improved_points_9),
                       ("37 knots", improved_points_37)]:
    plt.plot(times, distance_to_circle(points), label=label)
plt.xlabel("Time (s)")
plt.ylabel("Distance to circle (mm)")
plt.title("Problem 7 - Path Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
