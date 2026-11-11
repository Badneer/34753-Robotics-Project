import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp

from Part1.Problem3.main import q
from Part1.Problem4.main import J4, q1, q2, q3, q4


# Numerical Jacobian
J4_func = sp.lambdify((q1, q2, q3, q4), J4, "numpy")

# Knot configurations
indices = [0, 9, 18, 27, 36]
q_knots = q[indices]

# Desired Cartesian velocities (mm/s)
v_knots = np.array([
    [0,   0,   0],
    [0, -27,   0],
    [0,   0, -27],
    [0,  27,   0],
    [0,   0,   0]
])

# Calculate joint velocities at knots
qdot_knots = []

for qi, vi in zip(q_knots, v_knots):
    J = J4_func(*qi)

    J_reduced = np.vstack([
        J[:3, :],
        [0, 1, 1, 1]
    ])

    v = np.append(vi, 0)
    qdot_knots.append(np.linalg.solve(J_reduced, v))

qdot_knots = np.array(qdot_knots)


# Fifth-order polynomial coefficients
def polynomial_coefficients(q0, qf, dq0, dqf, T=2):
    a0 = q0
    a1 = dq0
    a2 = 0

    A = np.array([
        [T**3,   T**4,    T**5],
        [3*T**2, 4*T**3,  5*T**4],
        [6*T,    12*T**2, 20*T**3]
    ])

    b = np.array([
        qf - q0 - dq0*T,
        dqf - dq0,
        0
    ])

    a3, a4, a5 = np.linalg.solve(A, b)

    return np.array([a0, a1, a2, a3, a4, a5])


# Coefficients for 4 segments and 4 joints
coefficients = np.zeros((4, 4, 6))

for segment in range(4):
    for joint in range(4):
        coefficients[segment, joint] = polynomial_coefficients(
            q_knots[segment, joint],
            q_knots[segment + 1, joint],
            qdot_knots[segment, joint],
            qdot_knots[segment + 1, joint]
        )


# Complete trajectory (0 to 8 seconds)
def trajectory(t):
    if not 0 <= t <= 8:
        raise ValueError("Time must be between 0 and 8 seconds")

    segment = min(int(t // 2), 3)
    local_t = t - 2 * segment

    q_t = np.zeros(4)
    dq_t = np.zeros(4)
    ddq_t = np.zeros(4)

    for joint in range(4):
        c = coefficients[segment, joint]

        q_t[joint] = np.polynomial.polynomial.polyval(local_t, c)
        dq_t[joint] = np.polynomial.polynomial.polyval(
            local_t, np.polynomial.polynomial.polyder(c)
        )
        ddq_t[joint] = np.polynomial.polynomial.polyval(
            local_t, np.polynomial.polynomial.polyder(c, 2)
        )

    return q_t, dq_t, ddq_t


# Validation
if __name__ == "__main__":
    for segment in range(4):
        for joint in range(4):
            c = coefficients[segment, joint]

            dc = np.polynomial.polynomial.polyder(c)
            ddc = np.polynomial.polynomial.polyder(c, 2)

            assert np.isclose(
                np.polynomial.polynomial.polyval(0, c),
                q_knots[segment, joint]
            )
            assert np.isclose(
                np.polynomial.polynomial.polyval(2, c),
                q_knots[segment + 1, joint]
            )

            assert np.isclose(
                np.polynomial.polynomial.polyval(0, dc),
                qdot_knots[segment, joint]
            )
            assert np.isclose(
                np.polynomial.polynomial.polyval(2, dc),
                qdot_knots[segment + 1, joint]
            )

            assert np.isclose(
                np.polynomial.polynomial.polyval(0, ddc), 0
            )
            assert np.isclose(
                np.polynomial.polynomial.polyval(2, ddc), 0
            )

    print("All boundary conditions satisfied!")
    print("Coefficient array shape:", coefficients.shape)
    print("Joint velocities at knots (rad/s):")
    print(np.round(qdot_knots, 5))
