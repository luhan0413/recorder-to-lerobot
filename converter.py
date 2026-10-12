"""Raw v4/v5 to LeRobot v3: strict quality segments, CSV-time resampling, measured deltas."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from exporter import export_dataset, frames
from quality import Segment, build_segments, statistics
from schema import COLUMNS, DEFAULTS, FPS


@dataclass
class EpisodePlan:
    path: Path
    metadata: dict
    files: dict
    segment: Segment


def increasing(values, label, repeats=False):
    values = np.asarray(values, dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (np.diff(values) < 0 if repeats else np.diff(values) <= 0).any()
    ):
        raise ValueError(f"{label}: timestamps must be finite and increasing")
    return values


def camera_files(meta, path):
    cameras = meta.get("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError(
            f"{path}: cameras must map names to video filenames or objects with video_file"
        )
    result = {}
    for name, spec in cameras.items():
        if not name.isidentifier():
            raise ValueError(f"invalid camera name: {name}")
        if not isinstance(spec, (str, dict)):
            raise ValueError(f"{name}: camera must be a filename or object")
        filename = spec if isinstance(spec, str) else spec.get("video_file")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{name}: missing video_file")
        video = (path / filename).resolve()
        if not video.is_relative_to(path.resolve()):
            raise ValueError("video must be inside episode directory")
        result[name] = filename
    return result


def inspect_episode(path, required_cameras=("left",)):
    meta = json.loads((path / "episode.json").read_text())
    schema_version = meta.get(
        "raw_schema_version",
        meta.get("version") if meta.get("format") == "raw" else None,
    )
    if schema_version not in (4, 5):
        raise ValueError(f"{path}: expected raw schema version 4 or 5")
    if meta.get("status") != "completed" or meta.get("complete") is not True:
        raise ValueError(
            f"{path}: only completed episodes with complete=true are accepted"
        )
    if not isinstance(meta.get("task"), str) or not meta["task"].strip():
        raise ValueError(f"{path}: missing task")
    expected = {
        "position_unit": "meter",
        "rotation_unit": "radian",
        "rotation_type": "axis_angle",
        "pose_frame": "robot_base",
        "joint_unit": "radian",
        "gripper_source": "commanded",
        "gripper_zero": "fully_open",
        "gripper_one": "fully_closed",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{path}: must declare {key}={value}")
    with (path / "frames.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        if (reader.fieldnames or [])[: len(COLUMNS)] != COLUMNS:
            raise ValueError(f"{path}: base CSV columns must match schema v4 in order")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: duplicate CSV columns")
        rows = []
        for raw in reader:
            if None in raw:
                raise ValueError(f"{path}: CSV row has more values than columns")
            row = {
                key: (float(value) if value is not None and value.strip() else None)
                for key, value in raw.items()
            }
            if any(row[key] is None for key in COLUMNS) or any(
                v is not None and not math.isfinite(v) for v in row.values()
            ):
                raise ValueError(f"{path}: missing required value or nonfinite number")
            if row["frame_index"] != len(rows) or not 0 <= row["gripper_command"] <= 1:
                raise ValueError(f"{path}: invalid frame index or gripper command")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: at least two samples required")
    times = increasing([r["timestamp"] for r in rows], str(path))
    if abs(times[0]) > 1e-8:
        raise ValueError(f"{path}: first relative timestamp must be zero")
    if all(r.get("sample_host_monotonic") is not None for r in rows):
        host = np.array([r["sample_host_monotonic"] for r in rows])
        if not np.allclose(host - host[0], times, atol=1e-6, rtol=0):
            raise ValueError(
                f"{path}: timestamp differs from sample host relative time"
            )
    elif any(r.get("sample_host_monotonic") is not None for r in rows):
        raise ValueError(
            f"{path}: sample_host_monotonic must be present on every row or absent on every row"
        )
    files = camera_files(meta, path)
    if not set(required_cameras).issubset(files):
        raise ValueError(
            f"{path}: missing required cameras {sorted(set(required_cameras) - set(files))}"
        )
    if meta.get("visual_training_eligible") is False:
        raise ValueError(f"{path}: visual_training_eligible=false")
    shapes = {}
    for name in files:
        shape = None
        count = 0
        for frame in frames(path / files[name]):
            current = (frame.height, frame.width, 3)
            shape = shape or current
            if shape != current:
                raise ValueError(f"{path}: varying video dimensions")
            count += 1
        if count != len(rows):
            raise ValueError(
                f"{path}: {name} has {count} video frames but {len(rows)} CSV rows"
            )
        if shape is None:
            raise ValueError(f"{path}: empty video for {name}")
        if shape[0] % 2 or shape[1] % 2:
            raise ValueError("H.264 output needs even image dimensions")
        shapes[name] = shape
    return meta, rows, files, {name: shapes[name] for name in required_cameras}


def mapped_times(rows, column, mapping):
    """Apply an externally calibrated affine clock map. Never estimate exposure from receipts."""
    scale, offset = float(mapping["scale"]), float(mapping["offset"])
    if (
        not math.isfinite(scale)
        or scale <= 0
        or not math.isfinite(offset)
        or not mapping.get("provenance")
    ):
        raise ValueError(f"{column}: invalid calibration or missing provenance")
    if any(r.get(column) is None for r in rows):
        raise ValueError(f"{column}: calibrated mode requires all native timestamps")
    if "anchor_native_range" in mapping:
        low, high = mapping["anchor_native_range"]
        if (
            min(r[column] for r in rows) < low - 1e-8
            or max(r[column] for r in rows) > high + 1e-8
        ):
            raise ValueError(
                f"{column}: samples exceed calibrated anchor range; extrapolation refused"
            )
    return increasing([r[column] * scale + offset for r in rows], column, repeats=True)


def align_episode(
    path, rows, cameras, limits, clock_maps=None, timing_policy="sample-order"
):
    segments, report = build_segments(path, rows, cameras, limits)
    report.update(
        timing_policy=timing_policy,
        clock_mapping_applied=False,
        hardware_sync_verified=False,
    )
    # Native clock maps remain an optional diagnostic; CSV sampling time controls the new grid.
    if timing_policy == "calibrated":
        maps = (clock_maps or {}).get(path.name, {})
        required = ["rtde_controller_timestamp"] + [
            f"camera_{n}_sensor_timestamp" for n in cameras
        ]
        if any(column not in maps for column in required):
            raise ValueError(f"{path}: missing calibrated clock maps")
        arm = mapped_times(rows, required[0], maps[required[0]])
        report["calibrated_sensor_pairing_error"] = {
            n: statistics(
                abs(
                    mapped_times(
                        rows,
                        f"camera_{n}_sensor_timestamp",
                        maps[f"camera_{n}_sensor_timestamp"],
                    )
                    - arm
                )
            )
            for n in cameras
        }
        report["clock_mapping_applied"] = True
        report["clock_maps"] = maps
    return segments, report


def convert(
    source,
    output,
    repo_id,
    validate_only=False,
    cameras=("left",),
    timing_policy="sample-order",
    clock_maps=None,
    limits=None,
    report_path=None,
):
    limits = DEFAULTS | (limits or {})
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("camera names must be nonempty and unique")
    if timing_policy not in ("sample-order", "calibrated"):
        raise ValueError("unknown timing policy")
    if clock_maps is not None and timing_policy != "calibrated":
        raise ValueError("clock maps require calibrated timing policy")
    episodes = (
        [source]
        if (source / "episode.json").is_file()
        else sorted(p for p in source.glob("episode_*") if p.is_dir())
    )
    if not episodes:
        raise ValueError("no episode directories found")
    plans, baseline, source_reports, source_metadata = [], None, [], []
    for path in episodes:
        meta, rows, files, shapes = inspect_episode(path, cameras)
        signature = (meta.get("robot_type"), shapes)
        if not signature[0]:
            raise ValueError("missing robot_type")
        baseline = baseline or signature
        if signature != baseline:
            raise ValueError(
                "robot type and camera dimensions must match across episodes"
            )
        segments, report = align_episode(
            path, rows, cameras, limits, clock_maps, timing_policy
        )
        source_reports.append(report)
        source_metadata.append(meta)
        for segment in segments:
            segment.report["output_episode_index"] = len(plans)
            plans.append(EpisodePlan(path, meta, files, segment))
        print(
            f"Validated {path.name}: {len(rows)} raw -> {report['output_frames']} frames / {len(segments)} segments; "
            f"excluded={report['excluded_samples']}, unverifiable={report['unverifiable_samples']}"
        )
    quality_report = {
        "thresholds": {"target_fps": FPS, **limits},
        "episodes": source_reports,
        "output_episodes": len(plans),
        "output_frames": sum(len(p.segment.state) for p in plans),
        "excluded_samples": sum(r["excluded_samples"] for r in source_reports),
        "unverifiable_samples": sum(r["unverifiable_samples"] for r in source_reports),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as file:
            json.dump(quality_report, file, indent=2)
            file.write("\n")
    if validate_only:
        return source_reports
    if not plans:
        raise ValueError(
            "no valid segments with at least two output frames; inspect quality report"
        )
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    export_dataset(
        output,
        repo_id,
        baseline[0],
        baseline[1],
        plans,
        source_metadata,
        source_reports,
        quality_report,
        cameras,
        timing_policy,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset"))
    parser.add_argument("--repo-id", default="local/ur-dataset")
    parser.add_argument("--cameras", nargs="+", default=["left"])
    parser.add_argument(
        "--timing-policy", choices=["calibrated", "sample-order"], default=None
    )
    parser.add_argument("--clock-maps", type=Path)
    parser.add_argument(
        "--max-sample-gap-seconds",
        dest="max_gap",
        type=float,
        default=DEFAULTS["max_sample_gap_seconds"],
    )
    parser.add_argument(
        "--max-image-alignment-error-seconds",
        dest="max_alignment_error",
        type=float,
        default=DEFAULTS["max_image_alignment_error_seconds"],
    )
    parser.add_argument(
        "--max-rtde-age-seconds", type=float, default=DEFAULTS["max_rtde_age_seconds"]
    )
    parser.add_argument(
        "--max-camera-rtde-skew-seconds",
        type=float,
        default=DEFAULTS["max_camera_rtde_skew_seconds"],
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        maps = json.loads(args.clock_maps.read_text()) if args.clock_maps else None
        convert(
            args.input,
            args.output,
            args.repo_id,
            validate_only=args.validate_only,
            cameras=tuple(args.cameras),
            timing_policy=args.timing_policy
            or ("calibrated" if args.clock_maps else "sample-order"),
            clock_maps=maps,
            limits={
                "max_sample_gap_seconds": args.max_gap,
                "max_image_alignment_error_seconds": args.max_alignment_error,
                "max_rtde_age_seconds": args.max_rtde_age_seconds,
                "max_camera_rtde_skew_seconds": args.max_camera_rtde_skew_seconds,
            },
            report_path=args.report,
        )
    except (
        ValueError,
        KeyError,
        OSError,
        StopIteration,
        av.error.FFmpegError,
    ) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
