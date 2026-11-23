import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from Part1.Problem3.main import q, phi
from Part2.Problem6.main import trajectory, J4_func


def evaluate_path(configurations):
    """Calculate condition numbers, minimum singular values and ranks."""
    jacobians = np.array([J4_func(*angles) for angles in configurations])
    singular_values = np.linalg.svd(jacobians, compute_uv=False)
    conditions = np.linalg.cond(jacobians)
    minimum_values = singular_values[:, -1]
    ranks = np.linalg.matrix_rank(jacobians)
    return conditions, minimum_values, ranks


# Problem 3: all 37 elbow-up configurations on the exact circle.
condition_circle, minimum_circle, rank_circle = evaluate_path(q)

# Problem 6: the actual interpolated trajectory, sampled every 0.02 s.
times = np.linspace(0, 8, 401)
q_interpolated = np.array([trajectory(t)[0] for t in times])
condition_interpolated, minimum_interpolated, rank_interpolated = evaluate_path(
    q_interpolated
)


if __name__ == "__main__":
    print("Stylus geometric Jacobian: 6 rows, 4 columns; elbow-up branch.")
    print("Linear rows use mm; angular rows use radians.")
    print("Condition numbers depend on this unit scaling.\n")
    print("Path         Samples   Min. cond.   Max. cond.   Min. singular value   Min. rank")
    for name, conditions, minimum, ranks in [
        ("Problem 3", condition_circle, minimum_circle, rank_circle),
        ("Problem 6", condition_interpolated, minimum_interpolated, rank_interpolated),
    ]:
        print(f"{name:12s} {len(conditions):7d} {conditions.min():12.2f} "
              f"{conditions.max():12.2f} {minimum.min():21.4f} {ranks.min():11d}")

    if np.all(rank_circle == 4) and np.all(rank_interpolated == 4):
        print("\nNo singularities found at the sampled configurations.")
        print("Both paths retain full column rank (4).")
    else:
        print("\nA rank-deficient Jacobian was found along a sampled path.")
