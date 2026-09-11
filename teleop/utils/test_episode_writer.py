"""Hardware-free coverage for per-episode diagnostic metadata."""

import json
import tempfile
import unittest
from pathlib import Path

from teleop.utils.episode_writer import EpisodeWriter


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
