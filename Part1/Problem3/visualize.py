
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Part1.Problem3.main import points, positions_fk

plt.figure(figsize=(6, 6))

phi = np.linspace(0, 2*np.pi, 361)
plt.plot(32*np.cos(phi), 120 + 32*np.sin(phi), 'k--', label="Desired circle")
plt.scatter(positions_fk[:, 1], positions_fk[:, 2],
            s=20, label="Forward kinematics (elbow up)")
plt.scatter(points[0, 1], points[0, 2], marker="x", s=70, color="red",
            label="Start/end, phi = 0")

plt.xlabel("Y (mm)")
plt.ylabel("Z (mm)")
plt.title("Problem 3 - Circular Path Tracking")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
