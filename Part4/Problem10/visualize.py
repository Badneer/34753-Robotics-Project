import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from Part4.Problem10.main import (
    times, tau, inertia_torques, coriolis_torques, gravity_torques
)


# Each joint has its own scale so the small base-joint torque is visible.
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for joint, ax in enumerate(axes.flat):
    ax.plot(times, tau[:, joint], label="Total")
    ax.plot(times, gravity_torques[:, joint], "--", label="Gravity")
    ax.plot(times, inertia_torques[:, joint] + coriolis_torques[:, joint],
            ":", label="Inertia + Coriolis")
    ax.set_title(f"Joint {joint+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N m)")
    ax.grid(True)
    ax.legend(fontsize=8)

fig.suptitle("Problem 10 - Joint Torques along the Problem 6 Trajectory")
plt.tight_layout()
plt.show()
