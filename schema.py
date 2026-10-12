"""Shared raw columns, robot joints, and conversion defaults."""

POSE = ["x", "y", "z", "rx", "ry", "rz"]
JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
COLUMNS = (
    ["frame_index", "timestamp"]
    + [f"state_{n}" for n in POSE]
    + ["gripper_command"]
    + [f"joint_{i}" for i in range(1, 7)]
)


FPS = 10
DEFAULTS = {
    "max_sample_gap_seconds": 0.150,
    "max_image_alignment_error_seconds": 0.050,
    "max_rtde_age_seconds": 0.050,
    "max_camera_rtde_skew_seconds": 0.050,
}
