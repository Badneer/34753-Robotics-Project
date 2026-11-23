"""Problem 7: improve the circle approximation with 9 and 37 knots.

Keep the 8 s duration and quintic interpolation from Problem 6, but add
equally spaced circle points to shorten the segments. Interior velocities
are tangent to the circle with speed 2*pi*32/8 mm/s (about 25.1).
Start/end velocities and all knot joint accelerations are zero.
This follows the report; Problem 6 keeps its prescribed 27 mm/s velocities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp

from Part1.Problem1.main import T04, q1, q2, q3, q4
from Part1.Problem2.main import inverse_kinematics
from Part1.Problem4.main import J4
from Part2.Problem6.main import polynomial_coefficients


def circle_coefficients(n_knots):
    """Find quintic joint polynomials between n_knots circle points."""
    phi = np.linspace(0, 2*np.pi, n_knots)
    points = np.column_stack((
        np.full(n_knots, 150), 32*np.cos(phi), 120 + 32*np.sin(phi)
    ))
    q_knots = inverse_kinematics(points)
    speed = 2*np.pi*32 / 8
    v_knots = np.column_stack((
        np.zeros(n_knots), -speed*np.sin(phi), speed*np.cos(phi)
    ))
    v_knots[0] = 0
    v_knots[-1] = 0

    jacobian = sp.lambdify((q1, q2, q3, q4), J4, "numpy")
    qdot_knots = []
    for angles, velocity in zip(q_knots, v_knots):
        reduced = np.vstack((jacobian(*angles)[:3], [0, 1, 1, 1]))
        qdot_knots.append(np.linalg.solve(reduced, np.append(velocity, 0)))
    qdot_knots = np.array(qdot_knots)

    duration = 8 / (n_knots - 1)
    coefficients = np.zeros((n_knots - 1, 4, 6))
    for segment in range(n_knots - 1):
        for joint in range(4):
            coefficients[segment, joint] = polynomial_coefficients(
                q_knots[segment, joint], q_knots[segment + 1, joint],
                qdot_knots[segment, joint], qdot_knots[segment + 1, joint],
                T=duration
            )
    return coefficients


coefficients_9 = circle_coefficients(9)
coefficients_37 = circle_coefficients(37)


def improved_trajectory(t, coefficients=coefficients_9):
    """Return joint position, velocity and acceleration, as in Problem 6."""
    if not 0 <= t <= 8:
        raise ValueError("Time must be between 0 and 8 seconds")
    duration = 8 / len(coefficients)
    segment = min(int(t / duration), len(coefficients) - 1)
    local_t = t - segment * duration
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


# Same sampling interval as the baseline and the report: 0.02 s.
times = np.linspace(0, 8, 401)
fk = sp.lambdify((q1, q2, q3, q4), T04, "numpy")
improved_points_9 = np.array([
    fk(*improved_trajectory(t, coefficients_9)[0])[:3, 3] for t in times
])
improved_points_37 = np.array([
    fk(*improved_trajectory(t, coefficients_37)[0])[:3, 3] for t in times
])


if __name__ == "__main__":
    from Part2.Problem7.main import distance_to_circle

    print("More knots, same 8 s duration and quintic interpolation.")
    print(f"Interior knot speed: {2*np.pi*32/8:.4f} mm/s")
    print("Start/end at rest; zero joint acceleration at every knot.")
    for n_knots, points in [(9, improved_points_9), (37, improved_points_37)]:
        error = distance_to_circle(points)
        print(f"\n{n_knots} knots; segment duration {8/(n_knots-1):.4f} s")
        print(f"Maximum path error: {error.max():.6f} mm")
        print(f"Mean path error: {error.mean():.6f} mm")
