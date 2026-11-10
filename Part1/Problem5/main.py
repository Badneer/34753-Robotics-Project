import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from Part1.Problem3.main import q

from Part1.Problem4.main import J4, q1, q2, q3, q4
import sympy as sp

# Configuration at phi = pi/2
q_target = q[9]

# Convert symbolic Jacobian to numerical function
J4_func = sp.lambdify((q1, q2, q3, q4), J4, "numpy")

# Evaluate at phi = pi/2
J = J4_func(*q_target)

# Construct reduced 4x4 Jacobian
J_reduced = np.vstack([
    J[:3, :],
    [0, 1, 1, 1]
])

# Desired velocity
v = np.array([0, -3, 0, 0])

# Solve for joint velocities
q_dot = np.linalg.solve(J_reduced, v)

if __name__ == "__main__":
    print("Joint angles (degrees):")
    print(np.degrees(q_target))
    print("Reduced Jacobian:")
    print(np.round(J_reduced, 3))
    print("Joint velocities (rad/s):")
    print(q_dot)

    print("Joint velocities (deg/s):")
    print(np.degrees(q_dot))

    # Verify the result
    print("Calculated velocity:")
    print(J_reduced @ q_dot)