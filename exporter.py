"""Write validated conversion plans atomically with the official LeRobot writer."""

import json
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np

from schema import FPS, JOINTS, POSE


def frames(path):
    with av.open(str(path)) as container:
        yield from container.decode(video=0)


def selected_images(path, indices):
    """Decode sequentially, retaining only the current frame (supports repeated selections)."""
    stream = frames(path)
    current_index, current = -1, None
    try:
        for index in indices:
            while current_index < index:
                current = next(stream).to_ndarray(format="rgb24")
                current_index += 1
            yield current
    finally:
        stream.close()


def export_dataset(
    output,
    repo_id,
    robot_type,
    shapes,
    plans,
    source_metadata,
    source_reports,
    quality_report,
    cameras,
    timing_policy,
):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    joint_names = [name.removesuffix("_joint") for name in JOINTS] + ["gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": joint_names},
        "observation.tcp_pose": {"dtype": "float32", "shape": (7,), "names": POSE + ["gripper"]},
        "action_tcp": {
            "dtype": "float32", "shape": (7,),
            "names": [f"tcp_local_delta_{name}" for name in POSE] + ["gripper"],
        },
        "action_tcp_base": {
            "dtype": "float32", "shape": (7,),
            "names": [f"tcp_base_delta_{name}" for name in POSE] + ["gripper"],
        },
        "action_joints": {"dtype": "float32", "shape": (7,), "names": joint_names},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    features.update({
        f"observation.images.{name}": {
            "dtype": "video", "shape": shape, "names": ["height", "width", "channels"],
        }
        for name, shape in shapes.items()
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    root = staging / "dataset"
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=FPS,
            robot_type=robot_type,
            features=features,
            video_backend="pyav",
            image_writer_threads=8,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast"),
        )
        # Use the same episode filenames as the live v3 recording adapter.
        from lerobot.datasets.io_utils import write_info

        dataset.meta.info.data_path = "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet"
        dataset.meta.info.video_path = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4"
        write_info(dataset.meta.info, dataset.meta.root)
        for plan in plans:
            path, meta, files = plan.path, plan.metadata, plan.files
            aligned = plan.segment
            state, joints, action = aligned.state, aligned.joints, aligned.action
            selections = aligned.selections
            streams = {
                n: selected_images(path / files[n], selections[n]) for n in cameras
            }
            try:
                for i in range(len(state)):
                    data = {
                        "task": meta["task"],
                        "observation.state": joints[i],
                        "observation.tcp_pose": state[i],
                        **{key: values[i] for key, values in action.items()},
                        "next.done": np.array([i == len(state) - 1], dtype=bool),
                    }
                    data.update(
                        {f"observation.images.{n}": next(s) for n, s in streams.items()}
                    )
                    dataset.add_frame(data)
                dataset.save_episode(parallel_encoding=False)
            finally:
                for stream in streams.values():
                    stream.close()
        dataset.finalize()
        source_versions = sorted(
            {m.get("raw_schema_version", m.get("version")) for m in source_metadata}
        )
        provenance = {
            "raw_schema_version": source_versions[0]
            if len(source_versions) == 1
            else source_versions,
            "output_fps": FPS,
            "gripper_source": "commanded",
            "action_source": "next_measured_state",
            "state_source": "measured_joints_and_commanded_gripper",
            "action_spaces": {
                "action_tcp": "tcp_local_delta",
                "action_tcp_base": "tcp_base_delta",
                "action_joints": "next_absolute_joints",
            },
            "translation_frame": {"action_tcp": "current_tcp", "action_tcp_base": "robot_base"},
            "rotation_delta": {
                "action_tcp": "rotvec(R_current.T @ R_next)",
                "action_tcp_base": "rotvec(R_next @ R_current.T)",
            },
            "terminal_policy": "zero_tcp_deltas_current_joints_current_gripper_done_true",
            "alignment": "linear_position_and_joints_SO3_geodesic_rotation_nearest_image_previous_gripper_command",
            "timing_policy": timing_policy,
            "sensor_exposure_sync_verified": False,
            "thresholds": quality_report["thresholds"],
            "output_video": "H.264 CRF18 RGB-to-YUV420; raw input remains unchanged",
            "episodes": source_reports,
            "source_episode_metadata": source_metadata,
            "resampling_timebase": "CSV timestamp; MP4 PTS is never treated as capture time",
            "minimum_output_frames": 2,
        }
        images_dir = root / "images"
        if images_dir.exists():
            for directory in sorted(
                images_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True
            ):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            if not any(images_dir.iterdir()):
                images_dir.rmdir()
        (root / "meta" / "quality_report.json").write_text(
            json.dumps(quality_report, indent=2) + "\n"
        )
        (root / "meta" / "conversion.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        modality = {
            "state": {
                "joints": {"original_key": "observation.state", "start": 0, "end": 6},
                "gripper_position": {"original_key": "observation.state", "start": 6, "end": 7},
                "eef_position": {"original_key": "observation.tcp_pose", "start": 0, "end": 3},
                "eef_rotation": {"original_key": "observation.tcp_pose", "start": 3, "end": 6, "rotation_type": "axis_angle"},
            },
            "action": {},
            "video": {f"{n}_image": {"original_key": f"observation.images.{n}"} for n in cameras},
        }
        for key, frame in (("action_tcp", "current_tcp"), ("action_tcp_base", "robot_base")):
            modality["action"][key] = {
                "original_key": key, "start": 0, "end": 6,
                "absolute": False, "translation_frame": frame, "rotation_type": "axis_angle",
            }
            modality["action"][f"{key}_gripper"] = {
                "original_key": key, "start": 6, "end": 7, "absolute": True,
            }
        modality["action"]["action_joints"] = {
            "original_key": "action_joints", "start": 0, "end": 7, "absolute": True,
        }
        (root / "meta" / "modality.json").write_text(
            json.dumps(modality, indent=2) + "\n"
        )
        if output.exists():
            raise ValueError("output appeared during conversion; refusing overwrite")
        root.rename(output)
    finally:
        try:
            if dataset is not None and not dataset._is_finalized:
                dataset.finalize()
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    print(f"Exported LeRobot v3: {output}")
