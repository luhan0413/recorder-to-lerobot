"""Read back the schema-v4 example exported by the official writer."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def verify(root):
    dataset = LeRobotDataset(
        repo_id="local/ur-example", root=root, video_backend="pyav"
    )
    assert len(dataset) == 22 and dataset.num_episodes == 2
    for index in (0, 5, 10, 11, 21):
        row = dataset[index]
        local = index % 11
        np.testing.assert_allclose(
            float(row["observation.tcp_pose"][0]), local * 0.01, atol=1e-6
        )
        np.testing.assert_allclose(float(row["observation.state"][0]), local * 0.001, atol=1e-6)
        following = min(local + 1, 10)
        np.testing.assert_allclose(float(row["action_joints"][0]), following * 0.001, atol=1e-6)
        assert "action" not in row and "observation.joints" not in row
        assert "next.reward" not in row and "annotation.language.language_instruction" not in row
        assert int(row["frame_index"]) == local
        assert int(row["episode_index"]) == index // 11
        assert bool(row["next.done"]) == (local == 10)
        assert row["observation.images.arm"].shape == (3, 48, 64)
        if local == 10:
            np.testing.assert_allclose(row["action_tcp"][:6].numpy(), 0)
            np.testing.assert_allclose(row["action_tcp_base"][:6].numpy(), 0)
            np.testing.assert_allclose(row["action_joints"].numpy(), row["observation.state"].numpy())
    provenance = json.loads((root / "meta/conversion.json").read_text())
    assert provenance["action_source"] == "next_measured_state"
    assert provenance["gripper_source"] == "commanded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/example"))
    args = parser.parse_args()
    verify(args.root)
    print(
        "PASS: official reader, episodes, frames, poses, images, terminal action, provenance"
    )


if __name__ == "__main__":
    main()
