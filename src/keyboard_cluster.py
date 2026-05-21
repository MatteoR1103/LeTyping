from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .utils.tracking_utils import (
        activate_maintained_target_state,
        retrack_targets_from_current_frame,
    )
except ImportError:
    from utils.tracking_utils import (
        activate_maintained_target_state,
        retrack_targets_from_current_frame,
    )


@dataclass
class ClusterTargetPlan:
    target: dict
    current_letter: str
    immediate_next: dict | None
    key_position: np.ndarray
    track_during_hover: bool
    lock_key_position: bool
    next_requires_retrack: bool
    active_cluster: set[str]
    """
    Plan for handling the current target, including cluster state and retracking needs.
    - target: the current target dict
    - current_letter: the letter of the current target
    - immediate_next: the next target dict, or None if this is the last target
    - key_position: the world position of the current target key
    - track_during_hover: whether to keep tracking the target during the hover phase
    - lock_key_position: whether to lock the key position (no tracking) during hover
    - next_requires_retrack: whether the next target requires retracking before activation
    - active_cluster: the set of letters currently in the active tracking cluster
    """
    @property
    def should_go_home_after_press(self) -> bool:
        return self.next_requires_retrack or self.immediate_next is None


class KeyboardClusterManager:
    """Own the clustered-key state transitions used by the main typing loop."""

    def __init__(
        self,
        runtime_targets: list[dict],
        *,
        tracking_radius: float,
        min_distance: float,
        max_horizontal_delta: float,
        max_vertical_delta: float,
    ) -> None:
        self.runtime_targets = runtime_targets
        self.tracking_radius = tracking_radius
        self.min_distance = min_distance
        self.max_horizontal_delta = max_horizontal_delta
        self.max_vertical_delta = max_vertical_delta
        self.active_cluster: set[str] = set()
        self.frozen_world_by_letter: dict[str, np.ndarray] = {}
        self.retrack_from_home = True

    @classmethod
    def from_tracker(
        cls,
        tracker,
        letters: list[str],
        *,
        tracking_radius: float,
        min_distance: float,
        max_horizontal_delta: float,
        max_vertical_delta: float,
    ) -> "KeyboardClusterManager":
        runtime_targets = [dict(tracker.targets_by_letter[letter]) for letter in letters]
        return cls(
            runtime_targets,
            tracking_radius=tracking_radius,
            min_distance=min_distance,
            max_horizontal_delta=max_horizontal_delta,
            max_vertical_delta=max_vertical_delta,
        )

    def indexed_targets(self):
        return enumerate(self.runtime_targets)

    def prepare_target(
        self,
        index: int,
        target: dict,
        tracker,
        *,
        robot_interface,
        kinematics,
    ) -> ClusterTargetPlan:
        """
        Prepare the cluster state for the current target, returning a plan for how to handle it.
        Inputs:
            - index: index of the current target in the runtime_targets list
            - target: the current target dict
            - tracker: the current tracker object
            - robot_interface: the robot interface for setting targets and retracking
            - kinematics: the kinematics object for computing target states
        Returns:
            - ClusterTargetPlan: a dataclass containing the plan for handling the current target
        """
        immediate_next = (
            self.runtime_targets[index + 1]
            if index + 1 < len(self.runtime_targets)
            else None
        )
        current_letter = target["letter"]
        is_space_target = current_letter == "SPACE"
        cluster_candidates = self._cluster_candidates(index, is_space_target)

        if is_space_target:
            self.active_cluster = set()
            self.retrack_from_home = current_letter not in self.frozen_world_by_letter
        elif current_letter in self.frozen_world_by_letter:
            self.retrack_from_home = False

        if self.retrack_from_home:
            self._retrack_from_home(
                tracker,
                target,
                current_letter,
                cluster_candidates,
                robot_interface=robot_interface,
                kinematics=kinematics,
            )
        else:
            self._activate_current_target(
                tracker,
                target,
                current_letter,
                robot_interface=robot_interface,
                kinematics=kinematics,
            )

        next_requires_retrack = self._next_requires_retrack(immediate_next)
        key_position = np.asarray(target["world"], dtype=float).reshape(3)
        track_during_hover = bool(self.active_cluster)

        return ClusterTargetPlan(
            target=target,
            current_letter=current_letter,
            immediate_next=immediate_next,
            key_position=key_position,
            track_during_hover=track_during_hover,
            lock_key_position=not track_during_hover,
            next_requires_retrack=next_requires_retrack,
            active_cluster=set(self.active_cluster),
        )

    def finish_target(
        self,
        plan: ClusterTargetPlan,
        pressed_key_position: np.ndarray,
        tracker,
    ) -> None:
        if plan.active_cluster:
            self.frozen_world_by_letter[plan.current_letter] = np.asarray(
                pressed_key_position,
                dtype=float,
            ).reshape(3).copy()

            for letter in plan.active_cluster:
                if letter not in self.frozen_world_by_letter:
                    self.frozen_world_by_letter[letter] = np.asarray(
                        tracker.targets_by_letter[letter]["world"],
                        dtype=float,
                    ).reshape(3).copy()

            make_cluster_world_positions_coherent(
                plan.active_cluster,
                self.frozen_world_by_letter,
                plan.current_letter,
                min_dist_m=self.min_distance,
                max_horizontal_delta_m=self.max_horizontal_delta,
                max_vertical_delta_m=self.max_vertical_delta,
            )

        if plan.next_requires_retrack:
            self.active_cluster = set()
            self.retrack_from_home = True

    def _cluster_candidates(
        self,
        index: int,
        is_space_target: bool,
    ) -> list[str]:
        if is_space_target:
            return ["SPACE"]

        unrefined_remaining_letters = []
        for future_target in self.runtime_targets[index:]:
            letter = future_target["letter"]
            if letter not in self.frozen_world_by_letter and letter != "SPACE":
                unrefined_remaining_letters.append(letter)
        return unrefined_remaining_letters

    def _retrack_from_home(
        self,
        tracker,
        target: dict,
        current_letter: str,
        cluster_candidates: list[str],
        *,
        robot_interface,
        kinematics,
    ) -> None:
        retrack_targets_from_current_frame(
            tracker,
            cluster_candidates,
            robot_interface=robot_interface,
            kinematics=kinematics,
        )
        self.active_cluster = set(
            build_tracking_cluster(
                tracker.targets_by_letter,
                current_letter,
                cluster_candidates,
                radius=self.tracking_radius,
            )
        )
        tracker.active_cluster_letters = set(self.active_cluster)

        activate_maintained_target_state(tracker, current_letter)
        self.retrack_from_home = False

        if tracker.last_estimate is not None:
            target["world"] = tracker.last_estimate.copy()

    def _activate_current_target(
        self,
        tracker,
        target: dict,
        current_letter: str,
        *,
        robot_interface,
        kinematics,
    ) -> None:
        """
        Activate the current target without retracking, using frozen world positions if available.
        Inputs:
        - tracker: the current tracker object
        - target: the target dictionary
        - current_letter: the letter of the current target
        - robot_interface: the robot interface object
        - kinematics: the kinematics object
        """
        if current_letter in self.frozen_world_by_letter:
            self.active_cluster = set()
        tracker.active_cluster_letters = set(self.active_cluster)

        if current_letter in self.frozen_world_by_letter:
            frozen_world = self.frozen_world_by_letter[current_letter]
            activate_maintained_target_state(
                tracker,
                current_letter,
                world=frozen_world,
            )
            target["world"] = frozen_world.copy()
        else:
            tracker.set_target(
                letter=current_letter,
                robot_interface=robot_interface,
                kinematics=kinematics,
            )
            if tracker.last_estimate is not None:
                target["world"] = tracker.last_estimate.copy()

    def _next_requires_retrack(self, immediate_next: dict | None) -> bool:
        if immediate_next is None:
            return False

        immediate_next_letter = immediate_next["letter"]
        next_is_ready = (
            immediate_next_letter in self.active_cluster
            or immediate_next_letter in self.frozen_world_by_letter
        )
        next_requires_retrack = not next_is_ready

        return next_requires_retrack


def build_tracking_cluster(
    targets_by_letter: dict[str, dict],
    center_letter: str,
    candidate_letters: list[str],
    *,
    radius: float,
) -> list[str]:
    """
    Build the set of remaining letters close enough to track with the target.

    Distance is measured in world coordinates from `center_letter` using the
    latest estimates stored in `targets_by_letter`.
    """
    excluded_letters = ["SPACE"] # "P"
    if center_letter in excluded_letters:
        print(
            f"Tracking cluster around {center_letter} "
            f"(radius {radius:.3f} m): {center_letter}"
        )
        return [center_letter]



    center_world = np.asarray(
        targets_by_letter[center_letter]["world"],
        dtype=float,
    ).reshape(3)
    cluster: list[str] = []
    seen: set[str] = set()
    for letter in candidate_letters:
        if letter in seen:
            continue
        seen.add(letter)
        if letter == "SPACE":
            continue
        world = np.asarray(targets_by_letter[letter]["world"], dtype=float).reshape(3)
        distance = float(np.linalg.norm(world - center_world))
        if distance <= radius:
            cluster.append(letter)

    print(
        f"Tracking cluster around {center_letter} "
        f"(radius {radius:.3f} m): {', '.join(cluster)}"
    )
    return cluster


def make_cluster_world_positions_coherent(
    active_cluster: set[str],
    frozen_world_by_letter: dict[str, np.ndarray],
    anchor_letter: str,
    min_dist_m: float = 0.01,
    max_horizontal_delta_m: float = 0.022,
    max_vertical_delta_m: float = 0.014,
) -> None:
    """
    Adjust world positions of letters in the active cluster so they stay within
    the allowed component-wise offset band from anchor_letter.

    This prevents very close or identical key positions from causing problems
    during trajectory generation and execution.

    Inputs:
    - active_cluster: set of letters currently being tracked together
    - frozen_world_by_letter: dict mapping letters to frozen world positions
    - anchor_letter: letter in the cluster used as the reference point
    - min_dist_m: minimum allowed distance in metres between cluster letters
    - max_horizontal_delta_m: maximum allowed x-axis offset in metres
    - max_vertical_delta_m: maximum allowed y-axis offset in metres
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

        corrected = False
        if dist < min_dist_m:
            direction = np.array([1.0, 0.0, 0.0]) if dist < 1e-9 else delta / dist
            pos[:3] = anchor_pos[:3] + min_dist_m * direction
            delta = pos[:3] - anchor_pos[:3]
            corrected = True

        clamped_delta_xy = np.array(
            [
                float(np.clip(delta[0], -max_horizontal_delta_m, max_horizontal_delta_m)),
                float(np.clip(delta[1], -max_vertical_delta_m, max_vertical_delta_m)),
            ],
            dtype=float,
        )
        if not np.allclose(clamped_delta_xy, delta[:2]):
            pos[:2] = anchor_pos[:2] + clamped_delta_xy
            corrected = True

        if corrected:
            frozen_world_by_letter[letter] = pos
