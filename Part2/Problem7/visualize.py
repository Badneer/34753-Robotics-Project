
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from Part2.Problem7.main import actual_points
from Part2.Problem7.improve import improved_points

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
plt.plot(improved_points[:, 1], improved_points[:, 2],
         label="9 knots")

plt.xlabel("Y (mm)")
plt.ylabel("Z (mm)")
plt.title("Problem 7 - Cartesian Path Comparison")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()