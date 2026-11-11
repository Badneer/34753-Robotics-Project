import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from Part2.Problem6.main import trajectory


# Evaluate trajectory
times = np.linspace(0, 8, 401)

positions = []
velocities = []
accelerations = []

for t in times:
    q, dq, ddq = trajectory(t)

    positions.append(q)
    velocities.append(dq)
    accelerations.append(ddq)

positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)


# Plot joint trajectories
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

for joint in range(4):
    axes[0].plot(
        times, positions[:, joint], label=f"Joint {joint+1}"
    )
    axes[1].plot(
        times, velocities[:, joint], label=f"Joint {joint+1}"
    )
    axes[2].plot(
        times, accelerations[:, joint], label=f"Joint {joint+1}"
    )

axes[0].set_ylabel("Position (rad)")
axes[1].set_ylabel("Velocity (rad/s)")
axes[2].set_ylabel("Acceleration (rad/s²)")
axes[2].set_xlabel("Time (s)")

for ax in axes:
    ax.grid(True)
    ax.legend()

    # Knot boundaries
    for t in [2, 4, 6]:
        ax.axvline(
            t, color="gray", linestyle="--", alpha=0.5
        )

plt.tight_layout()
plt.show()
