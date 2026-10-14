"""Quality boundaries and official-writer round-trip checks."""

import ast
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


def recorder_reference():
    """Load only pure recorder functions, without hardware/service imports."""
    path = Path(__file__).resolve().parents[2] / "recording_training_replay_service.py"
    tree = ast.parse(path.read_text())
    functions = {"tcp_local_delta", "tcp_base_delta", "sample_to_lerobot_frame", "_lerobot_features"}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in functions or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("LEROBOT_") for t in node.targets)]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class QualityTests(unittest.TestCase):
    def test_actions_match_live_recorder(self):
        reference = recorder_reference()
        state = np.array([
            [0.2, -0.3, 0.4, 0.3, -0.5, 1.2, 0],
            [0.22, -0.29, 0.38, -0.4, 0.2, 1.3, 1],
            [0.21, -0.28, 0.42, 0.1, 0.6, -0.7, 0],
        ])
        joints = np.arange(21).reshape(3, 7) / 20
        joints[:, 6] = state[:, 6]
        calculated = actions(state, joints)
        for i in range(len(state)):
            following = min(i + 1, len(state) - 1)
            def sample(index):
                return {"tcp_pose": state[index, :6].tolist(),
                        "joints": joints[index, :6].tolist(),
                        "gripper": state[index, 6], "images": {}}
            expected = reference["sample_to_lerobot_frame"](
                sample(i), sample(following), "test", done=i == len(state) - 1)
            for key in calculated:
                np.testing.assert_allclose(calculated[key][i], expected[key], atol=1e-7)

    def test_preserve_episodes_keeps_gaps_and_diagnostic_outliers(self):
        rows = samples([0, 0.1, 0.4, 0.5])
        rows[1]["rtde_age_seconds"] = 0.2
        segments, report = build_segments(
            Path("episode_000000"), rows, ["left"], preserve_episodes=True
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(report["output_frames"], 6)
        self.assertEqual(report["excluded_samples"], 0)
        self.assertFalse(report["split_positions"])
        self.assertEqual(len(report["diagnostic_warnings"]), 1)
        self.assertTrue(report["alignment_threshold_exceedances"])
        np.testing.assert_allclose(segments[0].state[:, 0], np.arange(6) / 10)
        np.testing.assert_allclose(segments[0].action["action_tcp"][-1, :6], 0)

    def segments(self, rows, **limits):
        return build_segments(Path("episode_000000"), rows, ["left"], limits)

    def test_gap_splits_actions(self):
        segments, report = self.segments(samples([0, 0.1, 0.4, 0.5]))
        self.assertEqual(report["output_episodes"], 2)
        for segment in segments:
            np.testing.assert_allclose(segment.action["action_tcp"][-1, :6], 0)
            np.testing.assert_allclose(segment.action["action_tcp"][0, 0], 0.1)

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
        delta = actions(state)["action_tcp"]
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
        info = json.loads((self.output / "meta/info.json").read_text())
        self.assertEqual(info["data_path"], "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet")
        self.assertEqual(info["video_path"], "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4")
        expected = recorder_reference()["_lerobot_features"]({"left": np.zeros((48, 64, 3), dtype=np.uint8)})
        for key, feature in expected.items():
            self.assertEqual(info["features"][key]["dtype"], feature["dtype"])
            self.assertEqual(info["features"][key]["shape"], list(feature["shape"]))
            self.assertEqual(info["features"][key]["names"], feature["names"])
        self.assertEqual(set(info["features"]), set(expected) | {"timestamp", "frame_index", "episode_index", "index", "task_index"})
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
