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
            float(row["observation.state"][0]), local * 0.01, atol=1e-6
        )
        assert int(row["frame_index"]) == local
        assert int(row["episode_index"]) == index // 11
        assert bool(row["next.done"]) == (local == 10)
        assert row["observation.images.left"].shape == (3, 48, 64)
        if local == 10:
            np.testing.assert_allclose(row["action"][:6].numpy(), 0)
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
