"""Quality filtering and timestamp-authoritative resampling for raw recordings."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from schema import DEFAULTS, FPS, POSE


@dataclass
class Segment:
    state: np.ndarray
    joints: np.ndarray
    action: np.ndarray
    selections: dict
    report: dict


def statistics(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return {
        "count": len(values),
        "p95_seconds": float(np.percentile(values, 95)) if len(values) else None,
        "max_seconds": float(values.max()) if len(values) else None,
    }


def runs(mask):
    """Consecutive true indices; every false index establishes a boundary."""
    indices = np.flatnonzero(mask)
    return [
        part
        for part in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
        if len(part)
    ]


def nearest(times, grid):
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(grid - times[left]) <= abs(times[right] - grid), left, right)


def interpolate(rows, times, grid):
    pose = np.array([[row[f"state_{name}"] for name in POSE] for row in rows])
    joints = np.array([[row[f"joint_{i}"] for i in range(1, 7)] for row in rows])
    aligned = np.empty((len(grid), 6))
    aligned[:, :3] = np.column_stack(
        [np.interp(grid, times, pose[:, i]) for i in range(3)]
    )
    right = np.clip(np.searchsorted(times, grid, side="right"), 1, len(times) - 1)
    for k, (t, j) in enumerate(zip(grid, right)):
        fraction = (t - times[j - 1]) / (times[j] - times[j - 1])
        r0 = Rotation.from_rotvec(pose[j - 1, 3:])
        r1 = Rotation.from_rotvec(pose[j, 3:])
        delta = (r1 * r0.inv()).as_rotvec()
        aligned[k, 3:] = (Rotation.from_rotvec(delta * fraction) * r0).as_rotvec()
    aligned_joints = np.column_stack(
        [np.interp(grid, times, joints[:, i]) for i in range(6)]
    )
    previous = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(rows) - 1)
    grip = np.array([rows[i]["gripper_command"] for i in previous])
    return np.column_stack([aligned, grip]), np.column_stack([aligned_joints, grip])


def actions(state):
    result = np.zeros_like(state, dtype=np.float32)
    result[:-1, :3] = np.diff(state[:, :3], axis=0)
    rotation = Rotation.from_rotvec(state[:, 3:6])
    result[:-1, 3:6] = (rotation[1:] * rotation[:-1].inv()).as_rotvec()
    result[:-1, 6] = state[1:, 6]
    result[-1, 6] = state[-1, 6]
    return result


def build_segments(path, rows, cameras, limits=None):
    limits = DEFAULTS | (limits or {})
    for key, value in limits.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid threshold: {key}")
    times = np.array([row["timestamp"] for row in rows])
    gap = np.diff(times)
    report = {
        "episode": path.name,
        "raw_samples": len(rows),
        "excluded_samples": 0,
        "unverifiable_samples": 0,
        "excluded_rows": [],
        "split_positions": [],
        "discarded_fragments": [],
        "rejected_grid_points": [],
        "segments": [],
        "quality_verified": True,
        "max_sample_gap_seconds": float(gap.max()),
        "rtde_age": statistics([r.get("rtde_age_seconds") for r in rows]),
        "pairing_skew": {
            n: statistics([r.get(f"camera_{n}_pairing_skew_seconds") for r in rows])
            for n in cameras
        },
        "diagnostic_policy": "missing/negative age or host skew excludes row; native clocks are never subtracted",
    }
    valid = np.ones(len(rows), dtype=bool)
    diagnostics = [("rtde_age_seconds", limits["max_rtde_age_seconds"])] + [
        (f"camera_{n}_pairing_skew_seconds", limits["max_camera_rtde_skew_seconds"])
        for n in cameras
    ]
    for i, row in enumerate(rows):
        reasons, missing = [], []
        for column, threshold in diagnostics:
            value = row.get(column)
            if value is None:
                missing.append(column)
            elif value < 0:
                reasons.append(f"{column}:negative")
            elif value > threshold + 1e-9:
                reasons.append(f"{column}:above_limit")
        if missing or reasons:
            valid[i] = False
            report["excluded_rows"].append(
                {
                    "frame_index": i,
                    "timestamp": float(times[i]),
                    "reasons": reasons,
                    "missing_diagnostics": missing,
                }
            )
            report["split_positions"].append({"kind": "excluded_row", "frame_index": i})
            if missing:
                report["unverifiable_samples"] += 1
    report["excluded_samples"] = int((~valid).sum())
    report["quality_verified"] = report["unverifiable_samples"] == 0
    # Keep every original gap as a boundary, including gaps adjacent to excluded rows.
    gap_boundaries = np.flatnonzero(gap > limits["max_sample_gap_seconds"] + 1e-9) + 1
    for boundary in gap_boundaries:
        report["split_positions"].append(
            {
                "kind": "sample_gap",
                "before_frame_index": int(boundary),
                "gap_seconds": float(gap[boundary - 1]),
            }
        )
    chunks = []
    for part in runs(valid):
        chunks.extend(
            p
            for p in np.split(
                part, np.flatnonzero(np.isin(part[1:], gap_boundaries)) + 1
            )
            if len(p)
        )
    segments, all_errors = [], []
    for part in chunks:
        if len(part) < 2:
            report["discarded_fragments"].append(
                {
                    "source_frame_indices": part.tolist(),
                    "reason": "fewer_than_two_raw_samples",
                }
            )
            continue
        local_times = times[part]
        duration = local_times[-1] - local_times[0]
        grid = (
            local_times[0] + np.arange(int(np.floor(duration * FPS + 1e-8)) + 1) / FPS
        )
        selected = nearest(local_times, grid)
        error = abs(local_times[selected] - grid)
        adopt = error <= limits["max_image_alignment_error_seconds"] + 1e-9
        for i in np.flatnonzero(~adopt):
            rejected = {
                "timestamp": float(grid[i]),
                "nearest_frame_index": int(part[selected[i]]),
                "alignment_error_seconds": float(error[i]),
            }
            report["rejected_grid_points"].append(rejected)
            report["split_positions"].append({"kind": "image_alignment", **rejected})
        # Each adoption run becomes its own output episode. No action bridges a rejected grid.
        for adopted in runs(adopt):
            if len(adopted) < 2:
                report["discarded_fragments"].append(
                    {
                        "grid_timestamps": grid[adopted].tolist(),
                        "reason": "fewer_than_two_output_frames",
                    }
                )
                continue
            output_grid = grid[adopted]
            state, joint_state = interpolate(
                [rows[i] for i in part], local_times, output_grid
            )
            chosen = part[selected[adopted]]
            repeated = (len(chosen) - len(np.unique(chosen))) / len(chosen)
            segment_report = {
                "segment_index": len(segments),
                "output_frames": len(output_grid),
                "source_raw_span": [int(part[0]), int(part[-1])],
                "source_timestamp_start": float(output_grid[0]),
                "source_timestamp_end": float(output_grid[-1]),
                "source_grid_timestamps": output_grid.tolist(),
                "selected_video_frame_indices": {n: chosen.tolist() for n in cameras},
                "image_alignment_error": statistics(error[adopted]),
                "repeated_frame_ratio": {n: float(repeated) for n in cameras},
                "max_sample_gap_seconds": float(np.diff(local_times).max()),
                "rtde_age": statistics([rows[i].get("rtde_age_seconds") for i in part]),
                "pairing_skew": {
                    n: statistics(
                        [rows[i].get(f"camera_{n}_pairing_skew_seconds") for i in part]
                    )
                    for n in cameras
                },
            }
            all_errors.extend(error[adopted].tolist())
            report["segments"].append(segment_report)
            segments.append(
                Segment(
                    state.astype(np.float32),
                    joint_state.astype(np.float32),
                    actions(state),
                    {n: chosen for n in cameras},
                    segment_report,
                )
            )
    report["output_frames"] = sum(len(s.state) for s in segments)
    report["output_episodes"] = len(segments)
    report["image_alignment_error"] = statistics(all_errors)
    total = report["output_frames"]
    report["repeated_frame_ratio"] = {
        n: float(
            sum(
                len(s.selections[n]) - len(np.unique(s.selections[n])) for s in segments
            )
            / total
        )
        if total
        else None
        for n in cameras
    }
    return segments, report
