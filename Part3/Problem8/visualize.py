import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from Part3.Problem8.main import (
    phi, times, condition_circle, condition_interpolated
)


# Separate angle and time axes: the interpolated motion has varying speed.
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharey=True)

axes[0].plot(np.degrees(phi), condition_circle, "o-", markersize=3)
axes[0].set_xlabel("Circle angle (degrees)")
axes[0].set_title("Problem 3: exact circle (elbow up)")

axes[1].plot(times, condition_interpolated)
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Problem 6: interpolated trajectory (elbow up)")

for ax in axes:
    ax.set_ylabel("Condition number")
    ax.grid(True)

plt.tight_layout()
plt.show()
