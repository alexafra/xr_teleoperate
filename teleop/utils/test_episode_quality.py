"""Hardware-free tests for the read-only episode quality report."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from teleop.utils.episode_quality import classify_episode, main, scan_task


_LEFT_INSPIRE_NAMES = [
    "kLeftHandPinky",
    "kLeftHandRing",
    "kLeftHandMiddle",
    "kLeftHandIndex",
    "kLeftHandThumbBend",
    "kLeftHandThumbRotation",
]
_RIGHT_INSPIRE_NAMES = [name.replace("Left", "Right") for name in _LEFT_INSPIRE_NAMES]


def _timing(*, fps=30.0, max_gap=None, frame_count=3):
    sample_duration = 0.0 if frame_count < 2 else (frame_count - 1) / fps
    return {
        "target_fps": 30.0,
        "capture_start_utc": "2026-09-12T00:00:00+00:00",
        "capture_stop_utc": "2026-09-12T00:00:01+00:00",
        "recording_duration_s": sample_duration + 0.02,
        "sample_duration_s": sample_duration,
        "frame_count": frame_count,
        "measured_fps": fps,
        "max_frame_gap_s": (
            0.0 if frame_count < 2 else (1.0 / fps if max_gap is None else max_gap)
        ),
    }


def _dds(*, left=0, right=0, combined=None, protocol="dex3"):
    def stream(name, gap_count):
        if combined is not None:
            topic = "rt/inspire/state"
        elif protocol == "ftp":
            topic = f"rt/inspire_hand/state/{'l' if name == 'left' else 'r'}"
        else:
            topic = f"rt/dex3/{name}/state"
        return {
            "topic": topic,
            "has_received_valid_sample": True,
            "sample_count": 3,
            "gap_count": gap_count,
            "recovered_gap_count": gap_count,
            "open_gap_at_end": False,
            "last_sample_age_s_at_end": 0.001,
            "total_gap_duration_s": 0.08 * gap_count,
            "max_gap_duration_s": 0.08 if gap_count else 0.0,
            "gaps": [
                {
                    "start_offset_s": 0.01 + (0.1 * index),
                    "end_offset_s": 0.09 + (0.1 * index),
                    "duration_s": 0.08,
                    "recovered": True,
                }
                for index in range(gap_count)
            ],
        }

    if combined is not None:
        return {
            "schema_version": 1,
            "metric": "valid_state_receive_gap",
            "definition": (
                "A valid ChannelSubscriber.Read() inter-arrival gap strictly "
                "greater than gap_threshold_s"
            ),
            "gap_threshold_s": 0.075,
            "window_duration_s": 1.0,
            "combined": stream("combined", combined),
            "total_stream_gap_count": combined,
            "any_gap": combined > 0,
        }
    return {
        "schema_version": 1,
        "metric": "valid_state_receive_gap",
        "definition": (
            "A valid ChannelSubscriber.Read() inter-arrival gap strictly "
            "greater than gap_threshold_s"
        ),
        "gap_threshold_s": 0.075,
        "window_duration_s": 1.0,
        "left": stream("left", left),
        "right": stream("right", right),
        "total_side_gap_count": left + right,
        "any_gap": left + right > 0,
    }


def _dfx_lost(*, lost=0, resets=0, divergence=0, malformed=0):
    def side(side_lost, side_resets, side_divergence):
        baseline_count = 0
        held_count = baseline_count + int(side_lost > 0) + side_resets
        return {
            "sample_count": 3,
            "baseline_count": baseline_count,
            "accepted_count": 3 - held_count,
            "held_sample_count": held_count,
            "drop_event_count": int(side_lost > 0),
            "lost_increment_count": side_lost,
            "reset_count": side_resets,
            "counter_regression_count": side_resets - side_divergence,
            "counter_divergence_count": side_divergence,
            "baseline_at_start": [0] * 6,
            "baseline_at_end": [side_lost] * 6,
            "last_observed_counters": [side_lost] * 6,
            "events": [
                {"event": "held", "state_held": True} for _ in range(held_count)
            ],
        }

    left = side(lost, resets, divergence)
    right = side(0, 0, 0)
    total_held = left["held_sample_count"] + right["held_sample_count"]
    return {
        "schema_version": 1,
        "metric": "inspire_dfx_motor_state_lost_counter",
        "definition": (
            "Per-side DFX read failures inferred from six identical "
            "MotorState.lost counters; q is held on increments, counter "
            "regressions, divergent counters, and baseline samples"
        ),
        "window_duration_s": 1.0,
        "malformed_message_count": malformed,
        "malformed_messages": [
            {"offset_s": 0.1, "error": "bad sample"} for _ in range(malformed)
        ],
        "left": left,
        "right": right,
        "total_drop_event_count": int(lost > 0),
        "total_lost_increment_count": lost,
        "total_reset_count": resets,
        "total_held_sample_count": total_held,
        "any_state_held": total_held > 0,
        "any_anomaly": bool(lost or resets or malformed),
    }


def _inspire_contract(protocol):
    return {
        "schema_version": 1,
        "type": "inspire",
        "protocol": protocol,
        "hand_dof": 6,
        "value_unit": "normalized_open_fraction",
        "value_range": [0.0, 1.0],
        "zero_semantics": "fully_closed",
        "one_semantics": "fully_open",
        "left_joint_names": _LEFT_INSPIRE_NAMES,
        "right_joint_names": _RIGHT_INSPIRE_NAMES,
        "canonical_order": "left_then_right",
    }


def _inspire_info(protocol):
    return {
        "end_effector": _inspire_contract(protocol),
        "joint_names": {
            "left_ee": _LEFT_INSPIRE_NAMES,
            "right_ee": _RIGHT_INSPIRE_NAMES,
        },
    }


def _write_episode(task_dir, episode_name, document, *, add_data=True):
    episode_dir = task_dir / episode_name
    episode_dir.mkdir(parents=True)
    path = episode_dir / "data.json"
    document = dict(document)
    if add_data and "data" not in document:
        timing = document.get("timing", {})
        frame_count = timing.get("frame_count", 0)
        measured_fps = timing.get("measured_fps", 30.0)
        document["data"] = [
            {"idx": index, "timestamp_s": 0.01 + index / measured_fps}
            for index in range(frame_count)
        ]
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class TestEpisodeQualityClassification(unittest.TestCase):
    def test_dex3_is_inferred_and_clean_when_all_gaps_are_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0000",
                {
                    "timing": _timing(),
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                },
            )
            result = classify_episode(path)

        self.assertEqual(result.status, "clean")
        self.assertEqual(result.end_effector, "dex3")
        self.assertEqual(result.dds_gap_count, 0)

    def test_inspire_ftp_is_clean_with_both_zero_gap_streams(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0001",
                {
                    "info": _inspire_info("ftp"),
                    "timing": _timing(),
                    "diagnostics": {
                        "inspire_ftp_state_subscribers": _dds(protocol="ftp")
                    },
                },
            )
            result = classify_episode(path)

        self.assertEqual(result.status, "clean")
        self.assertEqual(result.end_effector, "inspire_ftp")
        self.assertIsNone(result.dfx_lost_increment_count)

    def test_inspire_dfx_is_clean_only_with_complete_zero_lost_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0002",
                {
                    "info": _inspire_info("dfx"),
                    "timing": _timing(),
                    "diagnostics": {
                        "inspire_dfx_state_subscribers": _dds(combined=0),
                        "inspire_dfx_lost_counters": _dfx_lost(),
                    },
                },
            )
            result = classify_episode(path)

        self.assertEqual(result.status, "clean")
        self.assertEqual(result.end_effector, "inspire_dfx")
        self.assertEqual(result.dfx_lost_increment_count, 0)
        self.assertEqual(result.dfx_reset_count, 0)
        self.assertEqual(result.dfx_divergence_count, 0)
        self.assertEqual(result.dfx_malformed_message_count, 0)

    def test_each_dfx_anomaly_rejects(self):
        anomaly_cases = (
            {"lost": 2},
            {"resets": 1},
            {"resets": 1, "divergence": 1},
            {"malformed": 1},
        )
        for index, anomaly in enumerate(anomaly_cases):
            with (
                self.subTest(anomaly=anomaly),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    {
                        "info": _inspire_info("dfx"),
                        "timing": _timing(),
                        "diagnostics": {
                            "inspire_dfx_state_subscribers": _dds(combined=0),
                            "inspire_dfx_lost_counters": _dfx_lost(**anomaly),
                        },
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "reject")
            self.assertTrue(any(reason.startswith("dfx:") for reason in result.reasons))

    def test_inspire_receive_gaps_reject_for_both_protocols(self):
        cases = (
            (
                "dfx",
                {
                    "inspire_dfx_state_subscribers": _dds(combined=1),
                    "inspire_dfx_lost_counters": _dfx_lost(),
                },
            ),
            (
                "ftp",
                {"inspire_ftp_state_subscribers": _dds(right=1, protocol="ftp")},
            ),
        )
        for index, (protocol, diagnostics) in enumerate(cases):
            with (
                self.subTest(protocol=protocol),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    {
                        "info": _inspire_info(protocol),
                        "timing": _timing(),
                        "diagnostics": diagnostics,
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "reject")
            self.assertGreater(result.dds_gap_count, 0)
            self.assertTrue(
                any(reason.startswith("dds_gap:") for reason in result.reasons)
            )

    def test_dds_gap_and_configured_timing_limits_reject(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0003",
                {
                    "timing": _timing(fps=28.5, max_gap=0.1),
                    "diagnostics": {"dex3_state_subscribers": _dds(left=1)},
                },
            )
            result = classify_episode(
                path,
                min_measured_fps=29.0,
                max_frame_gap_s=0.075,
            )

        self.assertEqual(result.status, "reject")
        self.assertTrue(any(reason.startswith("dds_gap:") for reason in result.reasons))
        self.assertTrue(
            any(reason.startswith("measured_fps=") for reason in result.reasons)
        )
        self.assertTrue(
            any(reason.startswith("max_frame_gap_s=") for reason in result.reasons)
        )

    def test_legacy_episode_without_diagnostics_is_unknown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0004",
                {"timing": _timing()},
            )
            result = classify_episode(path)

        self.assertEqual(result.status, "unknown")
        self.assertIn("diagnostics_missing", result.reasons)

    def test_hard_timing_failure_rejects_even_without_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0005",
                {"timing": _timing(fps=1.0, max_gap=2.0)},
            )
            result = classify_episode(path)

        self.assertEqual(result.status, "reject")
        self.assertIn("diagnostics_missing", result.reasons)

    def test_direct_episode_directory_uses_episode_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_episode(
                Path(temp_dir),
                "episode_0042",
                {
                    "timing": _timing(),
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                },
            )
            results = scan_task(path.parent)

        self.assertEqual(results[0].episode, "episode_0042")

    def test_incomplete_or_invalid_diagnostics_are_unknown(self):
        cases = (
            {"dex3_state_subscribers": {"available": False}},
            {"dex3_state_subscribers": {"left": {"gap_count": "zero"}}},
            {
                "inspire_dfx_state_subscribers": _dds(combined=0),
                "inspire_dfx_lost_counters": {"malformed_message_count": 0},
            },
        )
        for index, diagnostics in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp_dir:
                protocol = "dfx" if "inspire_dfx_lost_counters" in diagnostics else None
                document = {"timing": _timing(), "diagnostics": diagnostics}
                if protocol:
                    document["info"] = _inspire_info(protocol)
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    document,
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

    def test_missing_expected_dds_stream_is_unknown_for_each_hand_type(self):
        cases = []
        dex3 = _dds()
        dex3.pop("right")
        cases.append((None, {"dex3_state_subscribers": dex3}))
        ftp = _dds(protocol="ftp")
        ftp.pop("left")
        cases.append(("ftp", {"inspire_ftp_state_subscribers": ftp}))
        dfx = _dds(combined=0)
        dfx.pop("combined")
        cases.append(
            (
                "dfx",
                {
                    "inspire_dfx_state_subscribers": dfx,
                    "inspire_dfx_lost_counters": _dfx_lost(),
                },
            )
        )

        for index, (protocol, diagnostics) in enumerate(cases):
            with (
                self.subTest(protocol=protocol),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                document = {"timing": _timing(), "diagnostics": diagnostics}
                if protocol:
                    document["info"] = _inspire_info(protocol)
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    document,
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")
            self.assertTrue(
                any(
                    reason.startswith("dds_streams_missing:")
                    for reason in result.reasons
                )
            )

    def test_unobserved_or_zero_sample_dds_stream_is_unknown(self):
        for field, value in (
            ("has_received_valid_sample", False),
            ("sample_count", 0),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                dds = _dds()
                dds["left"][field] = value
                path = _write_episode(
                    Path(temp_dir),
                    "episode_0006",
                    {
                        "timing": _timing(),
                        "diagnostics": {"dex3_state_subscribers": dds},
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

    def test_unsupported_dds_schema_or_metric_is_unknown(self):
        for field, value in (
            ("schema_version", 2),
            ("metric", "different_metric"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                dds = _dds()
                dds[field] = value
                path = _write_episode(
                    Path(temp_dir),
                    "episode_0007",
                    {
                        "timing": _timing(),
                        "diagnostics": {"dex3_state_subscribers": dds},
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

    def test_incomplete_dfx_per_side_evidence_is_unknown(self):
        cases = []
        missing_side = _dfx_lost()
        missing_side.pop("right")
        cases.append(missing_side)
        missing_counter = _dfx_lost()
        missing_counter["left"].pop("counter_divergence_count")
        cases.append(missing_counter)
        wrong_metric = _dfx_lost()
        wrong_metric["metric"] = "different_metric"
        cases.append(wrong_metric)

        for index, lost_diagnostics in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp_dir:
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    {
                        "info": _inspire_info("dfx"),
                        "timing": _timing(),
                        "diagnostics": {
                            "inspire_dfx_state_subscribers": _dds(combined=0),
                            "inspire_dfx_lost_counters": lost_diagnostics,
                        },
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

    def test_frame_timestamps_are_required_and_cross_checked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mismatched = _write_episode(
                root,
                "episode_0010",
                {
                    "timing": _timing(),
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                },
            )
            mismatched_document = json.loads(mismatched.read_text(encoding="utf-8"))
            mismatched_document["data"][1]["timestamp_s"] += 0.01
            mismatched.write_text(json.dumps(mismatched_document), encoding="utf-8")

            missing = _write_episode(
                root,
                "episode_0011",
                {
                    "timing": _timing(),
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                },
            )
            missing_document = json.loads(missing.read_text(encoding="utf-8"))
            missing_document["data"][1].pop("timestamp_s")
            missing.write_text(json.dumps(missing_document), encoding="utf-8")

            mismatched_result = classify_episode(mismatched)
            missing_result = classify_episode(missing)

        self.assertEqual(mismatched_result.status, "reject")
        self.assertTrue(
            any("_mismatch:timing=" in reason for reason in mismatched_result.reasons)
        )
        self.assertEqual(missing_result.status, "unknown")
        self.assertTrue(
            any(
                reason.startswith("frame_timestamp_missing_or_invalid:")
                for reason in missing_result.reasons
            )
        )

    def test_timing_and_inspire_provenance_must_match_producer(self):
        cases = []
        missing_target = _timing()
        missing_target.pop("target_fps")
        cases.append((missing_target, _inspire_info("ftp")))

        boolean_schema = _inspire_info("ftp")
        boolean_schema["end_effector"]["schema_version"] = True
        cases.append((_timing(), boolean_schema))

        wrong_joint_names = _inspire_info("ftp")
        wrong_joint_names["joint_names"]["left_ee"] = ["wrong"] * 6
        cases.append((_timing(), wrong_joint_names))

        for index, (timing, info) in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp_dir:
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    {
                        "info": info,
                        "timing": timing,
                        "diagnostics": {
                            "inspire_ftp_state_subscribers": _dds(protocol="ftp")
                        },
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

    def test_dds_internal_accounting_cannot_false_clean(self):
        cases = []
        hidden_event = _dds()
        hidden_event["left"]["gaps"] = [
            {
                "start_offset_s": 0.1,
                "end_offset_s": 0.2,
                "duration_s": 0.1,
                "recovered": True,
            }
        ]
        cases.append(hidden_event)
        bad_open_flag = _dds()
        bad_open_flag["left"]["open_gap_at_end"] = True
        cases.append(bad_open_flag)
        missing_detail = _dds()
        missing_detail["left"].pop("recovered_gap_count")
        cases.append(missing_detail)

        for index, dds in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp_dir:
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    {
                        "timing": _timing(),
                        "diagnostics": {"dex3_state_subscribers": dds},
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

    def test_dfx_held_and_internal_accounting_cannot_false_clean(self):
        held = _dfx_lost()
        held["left"].update(
            {
                "baseline_count": 1,
                "accepted_count": 2,
                "held_sample_count": 1,
                "events": [{"event": "baseline", "state_held": True}],
            }
        )
        held["total_held_sample_count"] = 1
        held["any_state_held"] = True

        inconsistent = _dfx_lost()
        inconsistent["any_state_held"] = True

        for index, (lost_metrics, expected_status) in enumerate(
            ((held, "reject"), (inconsistent, "unknown"))
        ):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp_dir:
                path = _write_episode(
                    Path(temp_dir),
                    f"episode_{index:04d}",
                    {
                        "info": _inspire_info("dfx"),
                        "timing": _timing(),
                        "diagnostics": {
                            "inspire_dfx_state_subscribers": _dds(combined=0),
                            "inspire_dfx_lost_counters": lost_metrics,
                        },
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, expected_status)

    def test_conflicting_hand_sources_are_unknown_but_unrelated_gaps_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            conflicting = _write_episode(
                root,
                "episode_0012",
                {
                    "info": _inspire_info("ftp"),
                    "timing": _timing(),
                    "diagnostics": {
                        "inspire_ftp_state_subscribers": _dds(protocol="ftp"),
                        "inspire_dfx_lost_counters": _dfx_lost(),
                    },
                },
            )
            unrelated = _write_episode(
                root,
                "episode_0013",
                {
                    "timing": _timing(),
                    "diagnostics": {
                        "dex3_state_subscribers": _dds(),
                        "camera_health": {"gap_count": 9},
                    },
                },
            )

            conflicting_result = classify_episode(conflicting)
            unrelated_result = classify_episode(unrelated)

        self.assertEqual(conflicting_result.status, "unknown")
        self.assertTrue(
            any(
                reason.startswith("hand_diagnostic_source_set_mismatch:")
                for reason in conflicting_result.reasons
            )
        )
        self.assertEqual(unrelated_result.status, "clean")
        self.assertEqual(unrelated_result.dds_gap_count, 0)

    def test_frame_count_is_required_and_mismatch_rejects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = _write_episode(
                root,
                "episode_0007",
                {
                    "timing": {"measured_fps": 29.8, "max_frame_gap_s": 0.04},
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                    "data": [{}, {}, {}],
                },
            )
            mismatch = _write_episode(
                root,
                "episode_0008",
                {
                    "timing": _timing(frame_count=3),
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                    "data": [{}, {}],
                },
            )

            missing_result = classify_episode(missing)
            mismatch_result = classify_episode(mismatch)

        self.assertEqual(missing_result.status, "unknown")
        self.assertIn("frame_count_missing_or_invalid", missing_result.reasons)
        self.assertEqual(mismatch_result.status, "reject")
        self.assertTrue(
            any(
                reason.startswith("frame_count_mismatch:")
                for reason in mismatch_result.reasons
            )
        )


class TestEpisodeQualityCLI(unittest.TestCase):
    def test_scan_and_manifests_do_not_modify_episodes_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_path = _write_episode(
                root,
                "episode_0000",
                {
                    "timing": _timing(),
                    "diagnostics": {"dex3_state_subscribers": _dds()},
                },
            )
            unknown_path = _write_episode(
                root,
                "episode_0001",
                {"timing": _timing()},
            )
            before = {
                clean_path: clean_path.read_bytes(),
                unknown_path: unknown_path.read_bytes(),
            }
            json_manifest = root / "quality.json"
            tsv_manifest = root / "quality.tsv"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        str(root),
                        "--json-manifest",
                        str(json_manifest),
                        "--tsv-manifest",
                        str(tsv_manifest),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertIn("CLEAN 1", stdout.getvalue())
            manifest = json.loads(json_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["clean"], ["episode_0000"])
            self.assertEqual(manifest["unknown"], ["episode_0001"])
            self.assertIn(
                "status", tsv_manifest.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(scan_task(root)[0].status, "clean")
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)

            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                second_status = main([str(root), "--json-manifest", str(json_manifest)])
            self.assertEqual(second_status, 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
