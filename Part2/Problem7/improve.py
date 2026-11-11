import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from Part1.Problem3.main import inverse_kinematics

import sympy as sp

from Part1.Problem4.main import J4, q1, q2, q3, q4
from Part2.Problem6.main import polynomial_coefficients

# 9 knot points
phi = np.linspace(0, 2*np.pi, 9)

# Cartesian positions
points = np.column_stack((
    np.full(9, 150),
    32 * np.cos(phi),
    120 + 32 * np.sin(phi)
))

# Joint configurations
q_knots = inverse_kinematics(points)

print("Knot positions:", points.shape)
print("Joint configurations:", q_knots.shape)
print("Knot angles (degrees):", np.degrees(phi))

# Desired Cartesian velocities
v_knots = np.column_stack((
    np.zeros(9),
    -27 * np.sin(phi),
    27 * np.cos(phi)
))

v_knots[0] = 0
v_knots[-1] = 0

# Numerical Jacobian
J4_func = sp.lambdify((q1, q2, q3, q4), J4, "numpy")

# Calculate joint velocities
qdot_knots = []

for qi, vi in zip(q_knots, v_knots):
    J = J4_func(*qi)

    J_reduced = np.vstack((
        J[:3, :],
        [0, 1, 1, 1]
    ))

    qdot_knots.append(
        np.linalg.solve(J_reduced, np.append(vi, 0))
    )

qdot_knots = np.array(qdot_knots)

print("Joint velocities shape:", qdot_knots.shape)
print("Joint velocities (rad/s):")
print(np.round(qdot_knots, 5))


# 8 segments, 4 joints, 6 coefficients
coefficients = np.zeros((8, 4, 6))

for segment in range(8):
    for joint in range(4):
        coefficients[segment, joint] = polynomial_coefficients(
            q_knots[segment, joint],
            q_knots[segment + 1, joint],
            qdot_knots[segment, joint],
            qdot_knots[segment + 1, joint],
            T=1
        )


# Improved trajectory over 8 seconds
def improved_trajectory(t):
    if not 0 <= t <= 8:
        raise ValueError("Time must be between 0 and 8 seconds")

    segment = min(int(t // 1), 7)
    local_t = t - segment

    q_t = np.zeros(4)

    for joint in range(4):
        c = coefficients[segment, joint]
        q_t[joint] = np.polynomial.polynomial.polyval(local_t, c)

    return q_t


from Part1.Problem1.main import T04

# Numerical forward kinematics
fk = sp.lambdify((q1, q2, q3, q4), T04, "numpy")

times = np.linspace(0, 8, 801)

improved_points = []

for t in times:
    q_t = improved_trajectory(t)
    T = fk(*q_t)
    improved_points.append(T[:3, 3])

improved_points = np.array(improved_points)

# Calculate distance to desired circle
radial_error = np.abs(
    np.sqrt(improved_points[:, 1]**2 +
            (improved_points[:, 2] - 120)**2) - 32
)

x_error = np.abs(improved_points[:, 0] - 150)

path_error = np.sqrt(radial_error**2 + x_error**2)

if __name__ == "__main__":
    print("Improved trajectory:")
    print("Maximum radial error (mm):", np.max(radial_error))
    print("Maximum X deviation (mm):", np.max(x_error))
    print("Maximum path error (mm):", np.max(path_error))
