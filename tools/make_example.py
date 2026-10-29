"""Generate schema-v4 synthetic episodes (not real training demonstrations)."""

import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np

from schema import COLUMNS


def generate(output):
    if output.exists():
        raise ValueError("output already exists")
    clock_maps = {}
    for episode in range(2):
        root = output / f"episode_{episode:06d}"
        root.mkdir(parents=True)
        video_name = f"{root.name}.mp4"
        meta = {
            "raw_schema_version": 4,
            "status": "completed",
            "complete": True,
            "fps": 10,
            "interval": 0.1,
            "task": f"Synthetic task {episode}",
            "robot_type": "ur7e",
            "position_unit": "meter",
            "rotation_unit": "radian",
            "rotation_type": "axis_angle",
            "pose_frame": "robot_base",
            "joint_unit": "radian",
            "gripper_source": "commanded",
            "gripper_zero": "fully_open",
            "gripper_one": "fully_closed",
            "visual_training_eligible": True,
            "cameras": {"arm": video_name},
            "camera_specs": {
                "arm": {
                    "width": 64,
                    "height": 48,
                    "codec": "libx264rgb",
                    "pixel_format": "rgb24",
                }
            },
            "clock_mapping": None,
            "sensor_clock_calibration": False,
            "hardware_sync": False,
            "timing_columns": {
                "camera_arm_sensor_timestamp": {
                    "unit": "millisecond",
                    "clock": "camera",
                },
                "rtde_controller_timestamp": {"unit": "second", "clock": "controller"},
            },
            "pairing_rule": "abs(camera_arm_host_monotonic - rtde_host_monotonic)",
        }
        (root / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
        origin = 1000 + episode * 10
        times = np.arange(11) * 0.103
        extra = [
            "sample_host_monotonic",
            "capture_unix_timestamp",
            "rtde_controller_timestamp",
            "rtde_host_monotonic",
            "rtde_age_seconds",
            "camera_arm_sensor_timestamp",
            "camera_arm_host_monotonic",
            "camera_arm_pairing_skew_seconds",
        ]
        with (root / "frames.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS + extra)
            writer.writeheader()
            for i, t in enumerate(times):
                row = dict(
                    zip(
                        COLUMNS,
                        [
                            i,
                            t,
                            t * 0.1,
                            0.2,
                            0.3,
                            0,
                            0,
                            t * 0.2,
                            float(t >= 0.5),
                            *[j * 0.1 + t * 0.01 for j in range(6)],
                        ],
                    )
                )
                row.update(
                    sample_host_monotonic=origin + t,
                    capture_unix_timestamp=1791800000 + t,
                    rtde_controller_timestamp=50 + t,
                    rtde_host_monotonic=origin + t,
                    rtde_age_seconds=0,
                    camera_arm_sensor_timestamp=5000 + t * 1000,
                    camera_arm_host_monotonic=origin + t,
                    camera_arm_pairing_skew_seconds=0,
                )
                writer.writerow(row)
        with av.open(str(root / video_name), "w") as container:
            stream = container.add_stream("libx264rgb", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
            stream.options = {"crf": "0", "preset": "fast"}
            for i in range(len(times)):
                pixels = np.zeros((48, 64, 3), dtype=np.uint8)
                pixels[:, :, episode] = 50 + i * 15
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = i
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        clock_maps[root.name] = {
            "rtde_controller_timestamp": {
                "scale": 1,
                "offset": origin - 50,
                "provenance": "synthetic exact clock",
            },
            "camera_arm_sensor_timestamp": {
                "scale": 0.001,
                "offset": origin - 5,
                "provenance": "synthetic exact clock",
            },
        }
    (output / "clock_maps.json").write_text(json.dumps(clock_maps, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("raw_data/example"))
    args = parser.parse_args()
    try:
        generate(args.output)
    except ValueError as error:
        parser.error(str(error))
    print(args.output)


if __name__ == "__main__":
    main()
