import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp

from Part1.Problem2.main import inverse_kinematics
from Part1.Problem3.main import pc, R
from Part1.Problem4.main import J4, q1, q2, q3, q4


# Evaluate the exact circle every degree using the elbow-up branch.
phi = np.linspace(0, 2*np.pi, 361)
points = np.column_stack((
    np.full(len(phi), pc[0]),
    pc[1] + R*np.cos(phi),
    pc[2] + R*np.sin(phi)
))
q = inverse_kinematics(points)

# A 1 N force in the negative base-frame Z direction, with no applied moment.
force = np.array([0, 0, -1])
Jv_func = sp.lambdify((q1, q2, q3, q4), J4[:3, :], "numpy")

# Convert the linear Jacobian from mm to m so torques are in N m.
# The load torques and the motor torques required to hold the arm oppose.
tau_load = np.array([(1e-3 * Jv_func(*angles)).T @ force for angles in q])
tau = -tau_load


if __name__ == "__main__":
    print("External force: [0, 0, -1] N, expressed in the base frame.")
    print("Arm mass, friction and other losses are neglected.")
    print("Load torques: tau_load = Jv.T @ force, with Jv in m/rad.")
    print("Required holding torques: tau = -tau_load.\n")
    print("Holding torques (N m), positive in the joint-angle direction:")
    print("phi (deg)       tau1        tau2        tau3        tau4")
    for index in [0, 90, 180, 270, 360]:
        print(f"{index:9d} " + " ".join(f"{value:11.6f}" for value in tau[index]))
    print("\nJoint   Minimum (N m)   Maximum (N m)")
    for joint in range(4):
        print(f"{joint+1:5d} {tau[:, joint].min():15.6f} {tau[:, joint].max():15.6f}")
