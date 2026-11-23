import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp

from Part1.Problem1.main import A1, A2, A3, A4, q1, q2, q3, q4
from Part2.Problem6.main import trajectory


# All dynamics calculations use kg, m, s and radians.
angles = sp.Matrix([q1, q2, q3, q4])
velocities = sp.Matrix(sp.symbols("dq1 dq2 dq3 dq4", real=True))
masses = [sp.Rational(60, 1000), sp.Rational(80, 1000),
          sp.Rational(80, 1000), sp.Rational(40, 1000)]
gravity = sp.Rational(981, 100)

# Approximate link 1 as a solid prism: height 50 mm, depth 32 mm.
I0_estimate = float(masses[0]) * (0.050**2 + 0.032**2) / 12
I0 = sp.Rational(18, 1000000)  # Rounded estimate: 1.8e-5 kg m^2.
inertias = [
    I0 * sp.diag(1, sp.Rational(4, 10), sp.Rational(9, 10)),
    I0 * sp.diag(sp.Rational(45, 100), sp.Rational(14, 10), sp.Rational(12, 10)),
    I0 * sp.diag(sp.Rational(45, 100), sp.Rational(14, 10), sp.Rational(12, 10)),
    I0 * sp.diag(sp.Rational(5, 10), sp.Rational(5, 10), sp.Rational(5, 10)),
]

# Figure 2: local CoM offsets, with principal axes parallel to DH axes.
com_offsets = [sp.Matrix(p) / 1000 for p in
               [(0, -20, 0), (-30, 0, 0), (-30, 0, 0), (-25, 15, 0)]]

# Reuse the DH transformations, converting translations from mm to m.
transforms = [sp.eye(4)]
for link in [A1, A2, A3, A4]:
    link_m = link.copy()
    link_m[:3, 3] = link_m[:3, 3] / 1000
    transforms.append(sp.trigsimp(transforms[-1] * link_m))

D = sp.zeros(4)
potential = 0
com_positions = []
for i in range(4):
    rotation = transforms[i + 1][:3, :3]
    com = sp.trigsimp(rotation * com_offsets[i] + transforms[i + 1][:3, 3])
    com_positions.append(com)
    Jv = com.jacobian(angles)
    Jw = sp.Matrix.hstack(*[
        transforms[j][:3, 2] if j <= i else sp.zeros(3, 1)
        for j in range(4)
    ])
    D += masses[i] * Jv.T * Jv + Jw.T * rotation * inertias[i] * rotation.T * Jw
    potential += masses[i] * gravity * com[2]

D = sp.trigsimp(D)
g = sp.trigsimp(sp.Matrix([sp.diff(potential, angle) for angle in angles]))

# Christoffel formula: C[k,j] = sum_i c[i,j,k] * dq[i].
C = sp.zeros(4)
for k in range(4):
    for j in range(4):
        for i in range(4):
            C[k, j] += sp.Rational(1, 2) * (
                sp.diff(D[k, j], angles[i])
                + sp.diff(D[k, i], angles[j])
                - sp.diff(D[i, j], angles[k])
            ) * velocities[i]

D_func = sp.lambdify(tuple(angles), D, "numpy", cse=True)
C_func = sp.lambdify(tuple(angles) + tuple(velocities), C, "numpy", cse=True)
g_func = sp.lambdify(tuple(angles), g, "numpy", cse=True)

# Problem 6 trajectory; no additional 1 N tip load from Problem 9.
times = np.linspace(0, 8, 401)
inertia_torques = []
coriolis_torques = []
gravity_torques = []
for t in times:
    q, dq, ddq = trajectory(t)
    inertia_torques.append(D_func(*q) @ ddq)
    coriolis_torques.append(C_func(*q, *dq) @ dq)
    gravity_torques.append(g_func(*q).reshape(4))

inertia_torques = np.array(inertia_torques)
coriolis_torques = np.array(coriolis_torques)
gravity_torques = np.array(gravity_torques)
tau = inertia_torques + coriolis_torques + gravity_torques


if __name__ == "__main__":
    print(f"Prism estimate I0: {I0_estimate:.6e} kg m^2")
    print(f"Rounded I0 used: {float(I0):.6e} kg m^2")
    print("Model: D(q) ddq + C(q,dq) dq + g(q) = tau")
    print("Includes link masses and inertias; no extra tip load or friction.")
    print("\nSymbolic inertia matrix D(q):")
    sp.pprint(D)
    print("\nSymbolic gravity vector g(q):")
    sp.pprint(g)
    q, dq, ddq = trajectory(1.0)
    print("\nD, C and g at t = 1 s (SI units):")
    print("D =\n", D_func(*q))
    print("C =\n", C_func(*q, *dq))
    print("g =", g_func(*q).reshape(4))
    print("\nJoint   Minimum torque (N m)   Maximum torque (N m)")
    for joint in range(4):
        print(f"{joint+1:5d} {tau[:, joint].min():22.6f} {tau[:, joint].max():22.6f}")
    print("\nTime (s)       tau1        tau2        tau3        tau4 (N m)")
    for index in [0, 100, 200, 300, 400]:
        print(f"{times[index]:8.2f} " + " ".join(f"{value:11.6f}" for value in tau[index]))
