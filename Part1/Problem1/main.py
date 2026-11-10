
import sympy as sp

# Symbolic joint angles
q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4', real=True)

# Standard DH transformation
def dh(theta, d, a, alpha):
    c, s = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)

    return sp.Matrix([
        [c, -s*ca,  s*sa, a*c],
        [s,  c*ca, -c*sa, a*s],
        [0,     sa,    ca,   d],
        [0,      0,     0,   1]
    ])

# DH matrices
A1 = dh(q1, 50, 0, sp.pi/2)
A2 = dh(q2, 0, 93, 0)
A3 = dh(q3, 0, 93, 0)
A4 = dh(q4, 0, 50, 0)

# Stylus transformation
T04 = sp.trigsimp(A1 * A2 * A3 * A4)

# Camera offset
T45 = sp.eye(4)
T45[:3, 3] = sp.Matrix([-15, 45, 0])

# Camera transformation
T05 = sp.trigsimp(T04 * T45)

if __name__ == "__main__":
    print("T04:")
    sp.pprint(T04)

    print("\nT05:")
    sp.pprint(T05)