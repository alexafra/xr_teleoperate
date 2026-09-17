"""Hardware-free coverage for per-episode diagnostic metadata."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from teleop.utils.episode_writer import EpisodeWriter


def _depth_calibration():
    calibration = {
        "schema": "realsense_rgbd_calibration.v1",
        "camera": {
            "model": "Intel RealSense D435I",
            "serial": "254322071415",
            "product_id": "0B3A",
            "firmware": "5.15.1.55",
        },
        "color": {
            "width": 640,
            "height": 480,
            "fx": 609.3858642578125,
            "fy": 609.4705200195312,
            "cx": 325.95001220703125,
            "cy": 247.26507568359375,
            "distortion": "distortion.inverse_brown_conrady",
            "coeffs": [0.0] * 5,
            "format": "bgr8",
            "fps": 30,
        },
        "depth": {
            "width": 640,
            "height": 480,
            "fx": 397.5912170410156,
            "fy": 397.5912170410156,
            "cx": 315.6465148925781,
            "cy": 244.2028350830078,
            "distortion": "distortion.brown_conrady",
            "coeffs": [0.0] * 5,
            "format": "z16",
            "fps": 30,
        },
        "depth_to_color": {
            "rotation": [
                0.9999486207962036,
                0.0033009203616529703,
                0.009585974738001823,
                -0.0033925294410437346,
                0.9999485611915588,
                0.009556086733937263,
                -0.009553938172757626,
                -0.009588115848600864,
                0.9999083876609802,
            ],
            "translation_m": [
                0.014800711534917355,
                0.0008831368759274483,
                0.0007359444862231612,
            ],
        },
    }
    canonical_json = json.dumps(
        calibration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    calibration["fingerprint"] = (
        f"sha256:{hashlib.sha256(canonical_json).hexdigest()}"
    )
    return calibration


class _FakeDiagnosticsSource:
    def __init__(self, result):
        self.result = result
        self.begin_count = 0
        self.finish_count = 0

    def begin_episode_window(self):
        self.begin_count += 1

    def finish_episode_window(self):
        self.finish_count += 1
        return self.result


class TestEpisodeWriterDiagnostics(unittest.TestCase):
    def test_successful_episode_count_ignores_partial_directories_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(task_dir=str(task_dir), rerun_log=False)
            self.assertEqual(writer.get_successful_episode_count(), 0)
            self.assertTrue(writer.create_episode())
            self.assertEqual(writer.get_successful_episode_count(), 0)
            writer.close()
            self.assertEqual(writer.get_successful_episode_count(), 1)

            partial_dir = task_dir / "episode_9999"
            partial_dir.mkdir()
            (partial_dir / "data.json").write_text(
                '{\n"data": [\n{"idx": 0}',
                encoding="utf-8",
            )

            truncated_footer_dir = task_dir / "episode_9998"
            truncated_footer_dir.mkdir()
            (truncated_footer_dir / "data.json").write_text(
                '{\n"data": [],\n"timing": {"frame_count": 1',
                encoding="utf-8",
            )

            resumed = EpisodeWriter(task_dir=str(task_dir), rerun_log=False)
            try:
                self.assertEqual(resumed.get_successful_episode_count(), 1)
            finally:
                resumed.close()

    def test_depth_calibration_and_canonical_processing_scale_are_saved(self):
        calibration = _depth_calibration()

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(
                task_dir=str(task_dir),
                rerun_log=False,
                depth_scale_m_per_unit=0.0010000000474974513,
                depth_scale_reported_m_per_unit=0.0010000000474974513,
                depth_calibration=calibration,
            )
            calibration["camera"]["serial"] = "mutated"
            self.assertTrue(writer.create_episode())
            writer.close()
            saved = json.loads(
                (task_dir / "episode_0000" / "data.json").read_text()
            )

        self.assertEqual(saved["info"]["depth"]["scale_m_per_unit"], 0.001)
        self.assertEqual(
            saved["info"]["depth"]["scale_reported_m_per_unit"],
            0.0010000000474974513,
        )
        self.assertEqual(
            saved["info"]["depth"]["calibration"]["camera"]["serial"],
            "254322071415",
        )

    def test_depth_scale_requires_calibration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "depth_calibration is required"):
                EpisodeWriter(
                    task_dir=str(Path(temp_dir) / "episodes"),
                    rerun_log=False,
                    depth_scale_m_per_unit=0.001,
                )

    def test_noncanonical_depth_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "canonical"):
                EpisodeWriter(
                    task_dir=str(Path(temp_dir) / "episodes"),
                    rerun_log=False,
                    depth_scale_m_per_unit=0.0005,
                    depth_calibration=_depth_calibration(),
                )

    def test_tampered_calibration_fingerprint_is_rejected(self):
        calibration = _depth_calibration()
        calibration["color"]["fx"] += 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                EpisodeWriter(
                    task_dir=str(Path(temp_dir) / "episodes"),
                    rerun_log=False,
                    depth_scale_m_per_unit=0.001,
                    depth_calibration=calibration,
                )

    def test_rgbd_pairing_provenance_is_saved_per_frame(self):
        pairing = {
            "schema_version": 1,
            "mode": "atomic",
            "paired": True,
            "protocol": "teleimager-rgbd-v1",
            "capture_sequence": 42,
            "server_capture_monotonic_ns": 1234,
            "client_received_monotonic_ns": 5678,
            "raw_depth_pairing": "independent_legacy_stream",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(task_dir=str(task_dir), rerun_log=False)
            self.assertTrue(writer.create_episode())
            writer.add_item(colors={}, depths={}, rgbd_pairing=pairing)
            writer.close()
            saved = json.loads(
                (task_dir / "episode_0000" / "data.json").read_text()
            )

        self.assertEqual(saved["data"][0]["rgbd_pairing"], pairing)

    def test_legacy_call_omits_rgbd_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(task_dir=str(task_dir), rerun_log=False)
            self.assertTrue(writer.create_episode())
            writer.add_item(colors={}, depths={}, rgbd_pairing=None)
            writer.close()
            saved = json.loads(
                (task_dir / "episode_0000" / "data.json").read_text()
            )

        self.assertNotIn("rgbd_pairing", saved["data"][0])

    def test_dfx_receive_and_lost_counter_diagnostics_are_both_saved(self):
        receive_source = _FakeDiagnosticsSource(
            {"combined": {"gap_count": 0}}
        )
        lost_source = _FakeDiagnosticsSource(
            {
                "left": {"drop_event_count": 0},
                "right": {"drop_event_count": 2},
                "total_drop_event_count": 2,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(
                task_dir=str(task_dir),
                rerun_log=False,
                episode_diagnostics_sources={
                    "inspire_dfx_state_subscribers": receive_source,
                    "inspire_dfx_lost_counters": lost_source,
                },
            )
            self.assertTrue(writer.create_episode())
            writer.close()
            saved = json.loads((task_dir / "episode_0000" / "data.json").read_text())

        diagnostics = saved["diagnostics"]
        self.assertEqual(
            diagnostics["inspire_dfx_state_subscribers"]["combined"]["gap_count"],
            0,
        )
        self.assertEqual(
            diagnostics["inspire_dfx_lost_counters"]["total_drop_event_count"],
            2,
        )
        self.assertEqual(receive_source.begin_count, 1)
        self.assertEqual(receive_source.finish_count, 1)
        self.assertEqual(lost_source.begin_count, 1)
        self.assertEqual(lost_source.finish_count, 1)

    def test_save_freezes_diagnostics_before_async_finalize(self):
        payload = {"left": {"gap_count": 2}, "right": {"gap_count": 0}}
        source = _FakeDiagnosticsSource(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(
                task_dir=str(task_dir),
                rerun_log=False,
                episode_diagnostics_sources={"dex3_state_subscribers": source},
            )
            self.assertTrue(writer.create_episode())
            writer.save_episode()
            payload["left"]["gap_count"] = 99
            writer.close()

            saved = json.loads((task_dir / "episode_0000" / "data.json").read_text())

        self.assertEqual(source.begin_count, 1)
        self.assertEqual(source.finish_count, 1)
        self.assertEqual(
            saved["diagnostics"]["dex3_state_subscribers"]["left"]["gap_count"],
            2,
        )

    def test_close_finalizes_active_episode_with_diagnostics(self):
        source = _FakeDiagnosticsSource(
            {"left": {"gap_count": 0}, "right": {"gap_count": 1}}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(
                task_dir=str(task_dir),
                rerun_log=False,
                episode_diagnostics_sources={"dex3_state_subscribers": source},
            )
            self.assertTrue(writer.create_episode())
            writer.close()
            saved = json.loads((task_dir / "episode_0000" / "data.json").read_text())

        self.assertEqual(source.begin_count, 1)
        self.assertEqual(source.finish_count, 1)
        self.assertEqual(
            saved["diagnostics"]["dex3_state_subscribers"]["right"]["gap_count"],
            1,
        )

    def test_end_effector_contract_is_saved_and_populates_joint_names(self):
        end_effector_info = {
            "schema_version": 1,
            "type": "inspire",
            "protocol": "dfx",
            "hand_dof": 6,
            "value_unit": "normalized_open_fraction",
            "value_range": [0.0, 1.0],
            "zero_semantics": "fully_closed",
            "one_semantics": "fully_open",
            "left_joint_names": ["left_0", "left_1"],
            "right_joint_names": ["right_0", "right_1"],
            "canonical_order": "left_then_right",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "episodes"
            writer = EpisodeWriter(
                task_dir=str(task_dir),
                rerun_log=False,
                end_effector_info=end_effector_info,
            )
            end_effector_info["protocol"] = "mutated_after_construction"
            self.assertTrue(writer.create_episode())
            writer.close()
            saved = json.loads((task_dir / "episode_0000" / "data.json").read_text())

        self.assertEqual(saved["info"]["end_effector"]["protocol"], "dfx")
        self.assertEqual(saved["info"]["joint_names"]["left_ee"], ["left_0", "left_1"])
        self.assertEqual(saved["info"]["joint_names"]["right_ee"], ["right_0", "right_1"])


if __name__ == "__main__":
    unittest.main()
