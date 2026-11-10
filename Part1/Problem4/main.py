
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sympy as sp
from Part1.Problem1.main import A1, A2, A3, A4, T04, T05
from Part1.Problem1.main import q1, q2, q3, q4

import numpy as np
from Part1.Problem3.main import q

# Intermediate transformations
T00 = sp.eye(4)
T01 = A1
T02 = A1 * A2
T03 = T02 * A3

# Origins and rotation axes in base frame
T = [T00, T01, T02, T03]

origins = [Ti[:3, 3] for Ti in T]
axes = [Ti[:3, 2] for Ti in T]

def jacobian(T_end):
    o_end = T_end[:3, 3]

    columns = []
    for i in range(4):
        Jv = axes[i].cross(o_end - origins[i])
        Jw = axes[i]
        columns.append(Jv.col_join(Jw))

    return sp.trigsimp(sp.Matrix.hstack(*columns))


J4 = jacobian(T04)
J5 = jacobian(T05)

if __name__ == "__main__":
    print("Stylus Jacobian:", J4.shape)
    print("Camera Jacobian:", J5.shape)
    print("Stylus Jacobian (J4):")
    sp.pprint(J4)
    print("\nCamera Jacobian (J5):")
    sp.pprint(J5)

    # Convert symbolic Jacobians to numerical functions
    J4_func = sp.lambdify((q1, q2, q3, q4), J4, "numpy")
    J5_func = sp.lambdify((q1, q2, q3, q4), J5, "numpy")

    # Evaluate at the four requested points
    for i in [0, 9, 18, 27]:
        J4_num = J4_func(*q[i])
        J5_num = J5_func(*q[i])

        print(f"\nphi = {i*10} degrees")
        print("Stylus Jacobian:")
        print(np.round(J4_num, 3))
        print("Camera Jacobian:")
        print(np.round(J5_num, 3))
