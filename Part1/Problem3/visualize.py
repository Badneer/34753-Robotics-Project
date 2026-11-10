
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Part1.Problem3.main import points, positions_fk

plt.figure(figsize=(6, 6))

plt.plot(points[:, 1], points[:, 2], 'k--', label="Desired circle")
plt.scatter(positions_fk[:, 1], positions_fk[:, 2],
            s=20, label="Forward kinematics")

plt.xlabel("Y (mm)")
plt.ylabel("Z (mm)")
plt.title("Problem 3 - Circular Path Tracking")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
