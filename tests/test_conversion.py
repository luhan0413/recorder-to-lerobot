"""Quality boundaries and official-writer round-trip checks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from converter import convert, inspect_episode, mapped_times
from quality import actions, build_segments, interpolate
from tests.verify_example import verify
from tools.calibrate_clocks import fit_clock_maps
from tools.make_example import generate


def samples(times):
    return [
        dict(
            timestamp=t,
            gripper_command=float(t >= 0.15),
            rtde_age_seconds=0,
            camera_left_pairing_skew_seconds=0,
            **{
                f"state_{n}": t if n == "x" else 0
                for n in ("x", "y", "z", "rx", "ry", "rz")
            },
            **{f"joint_{j}": t for j in range(1, 7)},
        )
        for t in times
    ]


class QualityTests(unittest.TestCase):
    def segments(self, rows, **limits):
        return build_segments(Path("episode_000000"), rows, ["left"], limits)

    def test_gap_splits_actions(self):
        segments, report = self.segments(samples([0, 0.1, 0.4, 0.5]))
        self.assertEqual(report["output_episodes"], 2)
        for segment in segments:
            np.testing.assert_allclose(segment.action[-1, :6], 0)
            np.testing.assert_allclose(segment.action[0, 0], 0.1)

    def test_each_diagnostic_excludes_and_splits(self):
        for column in ("rtde_age_seconds", "camera_left_pairing_skew_seconds"):
            for value in (None, -0.001, 0.051):
                with self.subTest(column=column, value=value):
                    rows = samples([0, 0.1, 0.2, 0.3, 0.4])
                    rows[2][column] = value
                    segments, report = self.segments(rows)
                    self.assertEqual(len(segments), 2)
                    self.assertEqual(report["excluded_samples"], 1)
                    self.assertEqual(report["unverifiable_samples"], int(value is None))

    def test_rejected_grid_splits(self):
        segments, report = self.segments(
            samples([0, 0.1, 0.24, 0.34, 0.44]), max_image_alignment_error_seconds=0.01
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(report["rejected_grid_points"]), 3)

    def test_short_fragments_and_invalid_threshold(self):
        segments, report = self.segments(samples([0, 0.04]))
        self.assertFalse(segments)
        self.assertTrue(report["discarded_fragments"])
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.segments(samples([0, 0.1]), max_rtde_age_seconds=value)

    def test_rotation_and_gripper_interpolation(self):
        rows = samples([0, 0.2])
        rows[0]["state_rz"], rows[1]["state_rz"] = np.deg2rad([170, -170])
        state, _ = interpolate(rows, np.array([0, 0.2]), np.array([0, 0.1, 0.2]))
        np.testing.assert_allclose(abs(state[1, 5]), np.pi)
        np.testing.assert_allclose(state[:, 6], [0, 0, 1])
        delta = actions(state)
        np.testing.assert_allclose(delta[:-1, 5], np.deg2rad(10), atol=1e-6)
        np.testing.assert_allclose(delta[-1, :6], 0)

    def test_calibration_and_extrapolation(self):
        maps = fit_clock_maps(
            {
                "ep": {
                    "clock": {
                        "native": [1, 2, 3],
                        "common_host_monotonic": [11, 12, 13],
                        "provenance": "test anchors",
                    }
                }
            }
        )
        mapping = maps["ep"]["clock"]
        np.testing.assert_allclose(
            mapped_times([{"clock": 1}, {"clock": 3}], "clock", mapping), [11, 13]
        )
        with self.assertRaises(ValueError):
            mapped_times([{"clock": 4}], "clock", mapping)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.output = self.root / "dataset"
        generate(self.source)

    def convert(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return convert(self.source, self.output, "local/ur-example", **kwargs)

    def test_official_round_trip(self):
        maps = json.loads((self.source / "clock_maps.json").read_text())
        self.convert(clock_maps=maps, timing_policy="calibrated")
        verify(self.output)
        self.assertFalse(list(self.root.glob(".dataset-*")))
        with self.assertRaises(ValueError):
            self.convert()

    def test_validate_only_and_report(self):
        report = self.root / "quality.json"
        self.convert(validate_only=True, report_path=report)
        self.assertEqual(json.loads(report.read_text())["output_frames"], 22)
        self.assertFalse(self.output.exists())

    def test_incomplete_episode_and_frame_mismatch(self):
        episode = self.source / "episode_000000"
        path = episode / "episode.json"
        meta = json.loads(path.read_text())
        meta["complete"] = False
        path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            inspect_episode(episode)
        meta["complete"] = True
        path.write_text(json.dumps(meta))
        csv = episode / "frames.csv"
        csv.write_text("\n".join(csv.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaises(ValueError):
            inspect_episode(episode)

    def test_writer_failure_cleans_staging(self):
        with patch(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset.add_frame",
            side_effect=RuntimeError("write failed"),
        ):
            with self.assertRaises(RuntimeError):
                self.convert()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".dataset-*")))


if __name__ == "__main__":
    unittest.main()
