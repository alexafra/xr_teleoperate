"""Hardware-free tests for the read-only episode quality report."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from teleop.utils.episode_quality import classify_episode, main, scan_task


def _timing(*, fps=29.8, max_gap=0.04, frame_count=3):
    return {
        "frame_count": frame_count,
        "measured_fps": fps,
        "max_frame_gap_s": max_gap,
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
        }

    if combined is not None:
        return {
            "schema_version": 1,
            "metric": "valid_state_receive_gap",
            "combined": stream("combined", combined),
            "total_stream_gap_count": combined,
        }
    return {
        "schema_version": 1,
        "metric": "valid_state_receive_gap",
        "left": stream("left", left),
        "right": stream("right", right),
        "total_side_gap_count": left + right,
    }


def _dfx_lost(*, lost=0, resets=0, divergence=0, malformed=0):
    def side(side_lost, side_resets, side_divergence):
        return {
            "drop_event_count": int(side_lost > 0),
            "lost_increment_count": side_lost,
            "reset_count": side_resets,
            "counter_regression_count": side_resets - side_divergence,
            "counter_divergence_count": side_divergence,
        }

    return {
        "schema_version": 1,
        "metric": "inspire_dfx_motor_state_lost_counter",
        "malformed_message_count": malformed,
        "left": side(lost, resets, divergence),
        "right": side(0, 0, 0),
        "total_drop_event_count": int(lost > 0),
        "total_lost_increment_count": lost,
        "total_reset_count": resets,
    }


def _inspire_contract(protocol):
    return {"type": "inspire", "protocol": protocol}


def _write_episode(task_dir, episode_name, document, *, add_data=True):
    episode_dir = task_dir / episode_name
    episode_dir.mkdir(parents=True)
    path = episode_dir / "data.json"
    document = dict(document)
    if add_data and "data" not in document:
        timing = document.get("timing", {})
        frame_count = timing.get("frame_count", 0)
        document["data"] = [{} for _ in range(frame_count)]
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
                    "info": {"end_effector": _inspire_contract("ftp")},
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
                    "info": {"end_effector": _inspire_contract("dfx")},
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
                        "info": {"end_effector": _inspire_contract("dfx")},
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
                        "info": {"end_effector": _inspire_contract(protocol)},
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
                    document["info"] = {"end_effector": _inspire_contract(protocol)}
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
                    document["info"] = {"end_effector": _inspire_contract(protocol)}
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
                        "info": {"end_effector": _inspire_contract("dfx")},
                        "timing": _timing(),
                        "diagnostics": {
                            "inspire_dfx_state_subscribers": _dds(combined=0),
                            "inspire_dfx_lost_counters": lost_diagnostics,
                        },
                    },
                )
                result = classify_episode(path)

            self.assertEqual(result.status, "unknown")

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
