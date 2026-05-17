from __future__ import annotations

import numpy as np


def make_cluster_world_positions_coherent(
    active_cluster: set[str],
    frozen_world_by_letter: dict[str, np.ndarray],
    anchor_letter: str,
    min_dist_m: float = 0.01,
) -> None:
    """
    Adjust world positions of letters in the active cluster so they are at
    least min_dist_m apart, using anchor_letter as the reference point.

    This prevents very close or identical key positions from causing problems
    during trajectory generation and execution.

    Inputs:
    - active_cluster: set of letters currently being tracked together
    - frozen_world_by_letter: dict mapping letters to frozen world positions
    - anchor_letter: letter in the cluster used as the reference point
    - min_dist_m: minimum allowed distance in metres between cluster letters
    """
    if anchor_letter not in frozen_world_by_letter:
        return

    anchor_pos = np.asarray(frozen_world_by_letter[anchor_letter], dtype=float).copy()
    for letter in sorted(active_cluster):
        if letter == anchor_letter or letter not in frozen_world_by_letter:
            continue

        pos = np.asarray(frozen_world_by_letter[letter], dtype=float).copy()
        delta = pos[:3] - anchor_pos[:3]
        dist = float(np.linalg.norm(delta))

        if dist >= min_dist_m:
            continue

        direction = np.array([1.0, 0.0, 0.0]) if dist < 1e-9 else delta / dist
        pos[:3] = anchor_pos[:3] + min_dist_m * direction
        frozen_world_by_letter[letter] = pos
        print(
            f"[WARNING] Corrected collapsed key positions {anchor_letter}-{letter}: "
            f"distance was {dist * 1000:.2f} mm, enforced {min_dist_m * 1000:.1f} mm."
        )
