import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from Part1.Problem1.main import T04

o04 = T04[:3, 3]  # Stylus position
x04 = T04[:3, 0]  # Stylus x-axis orientation


def inverse_kinematics(points, x_z=0, elbow_up=True,
                       forward=True, supplementary=False):
    """Satisfy the tip position (mm) and the vertical stylus-axis component.

    Return one row of joint angles (rad) per point. The defaults select
    the forward-reaching, elbow-up branch with psi = arcsin(x_z).
    Other branches can be selected with the three boolean arguments.
    Joint limits are not imposed by this geometric solution.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Points must contain finite (x, y, z) coordinates")
    if not np.isfinite(x_z) or abs(x_z) > 1:
        raise ValueError("x_z must be between -1 and 1")

    x, y, z = points.T
    base = np.arctan2(y, x)
    reach = np.hypot(x, y)
    if not forward:
        base = base + np.pi
        reach = -reach

    psi = np.arcsin(x_z)
    if supplementary:
        psi = np.pi - psi

    # Step back from the stylus tip to joint 4 in the arm plane.
    wrist_r = reach - 50 * np.cos(psi)
    wrist_z = z - 50 - 50 * np.sin(psi)
    cos_elbow = (wrist_r**2 + wrist_z**2 - 2 * 93**2) / (2 * 93**2)
    if np.any(np.abs(cos_elbow) > 1 + 1e-10):
        raise ValueError("Unreachable target position for the selected branch")

    elbow = np.arccos(np.clip(cos_elbow, -1, 1))
    if elbow_up:
        elbow = -elbow
    shoulder = np.arctan2(wrist_z, wrist_r) - np.arctan2(
        93 * np.sin(elbow), 93 + 93 * np.cos(elbow)
    )
    wrist = psi - shoulder - elbow
    return np.column_stack((base, shoulder, elbow, wrist))


if __name__ == "__main__":
    print("Inverse kinematics (lengths in mm, angles in rad):")
    print("r = +/-sqrt(x^2 + y^2); q1 = atan2(y, x) (+ pi for backward reach)")
    print("psi = asin(x_z) or pi - asin(x_z)")
    print("rw = r - 50*cos(psi); zw = z - 50 - 50*sin(psi)")
    print("D = (rw^2 + zw^2 - 2*93^2)/(2*93^2); require |D| <= 1")
    print("q3 = +/-acos(D); negative for elbow up")
    print("q2 = atan2(zw, rw) - atan2(93*sin(q3), 93 + 93*cos(q3))")
    print("q4 = psi - q2 - q3")
    print("\nExample: tip = [150, 32, 120] mm, x_z = 0, elbow up")
    print(inverse_kinematics([150, 32, 120])[0])
