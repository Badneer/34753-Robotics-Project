import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from Part3.Problem9.main import phi, tau


plt.figure(figsize=(8, 5))
for joint in range(4):
    plt.plot(np.degrees(phi), tau[:, joint], label=f"Joint {joint+1}")

plt.xlabel("Circle angle (degrees)")
plt.ylabel("Holding torque (N m)")
plt.title("Problem 9 - Static Torques for a Downward 1 N Load")
plt.xticks([0, 90, 180, 270, 360])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
