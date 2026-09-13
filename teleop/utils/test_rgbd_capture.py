"""Hardware-free tests for automatic atomic RGBD recording selection."""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path

from teleop.utils.rgbd_capture import (
    EXPERIMENTAL_ATOMIC_RGBD_RECORDING,
    PreferAtomicHeadRgbdCapture,
    should_prefer_atomic_rgbd_recording,
)


class _Clock:
    def __init__(self):
        self.now_ns = 10_000_000_000

    def __call__(self):
        return self.now_ns

    def advance(self, seconds):
        self.now_ns += int(seconds * 1_000_000_000)


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class _LegacyImage:
    def __init__(self, bgr):
        self.bgr = bgr


@dataclass
class _AtomicFrame:
    sequence: int
    server_capture_monotonic_ns: int
    received_monotonic_ns: int
    color: object
    depth: object


class _ImageClient:
    def __init__(self, atomic_values):
        self.atomic_values = list(atomic_values)
        self.atomic_calls = 0
        self.legacy_color_calls = 0
        self.legacy_depth_calls = 0

    def get_head_rgbd_frame(self):
        self.atomic_calls += 1
        if not self.atomic_values:
            return None
        value = self.atomic_values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def get_head_frame(self):
        self.legacy_color_calls += 1
        return _LegacyImage("legacy-color")

    def get_head_depth_frame(self):
        self.legacy_depth_calls += 1
        return "legacy-depth"


def _head_config():
    return {
        "enable_depth": True,
        "rgbd_zmq_port": 5560,
        "rgbd_protocol": "teleimager-rgbd-v1",
    }


def _decode(frame):
    return frame.color, frame.depth


def _frame(clock, sequence):
    return _AtomicFrame(
        sequence=sequence,
        server_capture_monotonic_ns=1_000 + sequence,
        received_monotonic_ns=clock(),
        color=f"atomic-color-{sequence}",
        depth=f"atomic-depth-{sequence}",
    )


class TestPreferAtomicHeadRgbdCapture(unittest.TestCase):
    def test_policy_is_limited_to_inspire_depth_recording(self):
        self.assertFalse(EXPERIMENTAL_ATOMIC_RGBD_RECORDING)
        for end_effector in ("inspire_dfx", "inspire_ftp"):
            self.assertFalse(
                should_prefer_atomic_rgbd_recording(
                    recording_enabled=True,
                    end_effector=end_effector,
                    head_config={"enable_depth": True},
                )
            )
            self.assertTrue(
                should_prefer_atomic_rgbd_recording(
                    recording_enabled=True,
                    end_effector=end_effector,
                    head_config={"enable_depth": True},
                    experimental_opt_in=True,
                )
            )

        for end_effector in ("dex1", "dex3", "brainco", None):
            self.assertFalse(
                should_prefer_atomic_rgbd_recording(
                    recording_enabled=True,
                    end_effector=end_effector,
                    head_config={"enable_depth": True},
                    experimental_opt_in=True,
                )
            )
        self.assertFalse(
            should_prefer_atomic_rgbd_recording(
                recording_enabled=False,
                end_effector="inspire_ftp",
                head_config={"enable_depth": True},
                experimental_opt_in=True,
            )
        )
        self.assertFalse(
            should_prefer_atomic_rgbd_recording(
                recording_enabled=True,
                end_effector="inspire_ftp",
                head_config={"enable_depth": False},
                experimental_opt_in=True,
            )
        )

    def test_teleop_constructor_is_guarded_by_inspire_policy(self):
        source_path = Path(__file__).parents[1] / "teleop_hand_and_arm.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PreferAtomicHeadRgbdCapture"
        ]

        self.assertEqual(len(constructors), 1)
        node = constructors[0]
        guards = []
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                guards.append(ast.unparse(node.test))
        self.assertTrue(
            any("should_prefer_atomic_rgbd_recording" in guard for guard in guards)
        )
        policy_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "should_prefer_atomic_rgbd_recording"
        ]
        self.assertEqual(len(policy_calls), 1)
        opt_in = {
            keyword.arg: keyword.value for keyword in policy_calls[0].keywords
        }["experimental_opt_in"]
        self.assertIsInstance(opt_in, ast.Name)
        self.assertEqual(opt_in.id, "EXPERIMENTAL_ATOMIC_RGBD_RECORDING")

    def test_teleop_does_not_gate_episode_rows_on_atomic_frame_arrival(self):
        source_path = Path(__file__).parents[1] / "teleop_hand_and_arm.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        self.assertFalse(
            any(
                isinstance(node, ast.Name)
                and node.id == "rgbd_sample_ready"
                for node in ast.walk(tree)
            )
        )

    def test_prefers_atomic_and_records_pairing_provenance(self):
        clock = _Clock()
        client = _ImageClient([_frame(clock, 7)])
        capture = PreferAtomicHeadRgbdCapture(
            client,
            _head_config(),
            decode_atomic_frame=_decode,
            logger=_Logger(),
            monotonic_ns=clock,
        )

        sample = capture.read()

        self.assertEqual(sample.color_bgr, "atomic-color-7")
        self.assertEqual(sample.aligned_depth, "atomic-depth-7")
        self.assertEqual(sample.pairing["mode"], "atomic")
        self.assertTrue(sample.pairing["paired"])
        self.assertEqual(sample.pairing["capture_sequence"], 7)
        self.assertEqual(client.legacy_color_calls, 0)
        self.assertEqual(client.legacy_depth_calls, 0)
        self.assertEqual(
            capture.episode_metadata()["raw_depth_pairing"],
            "independent_legacy_stream",
        )

    def test_receive_during_getter_is_not_misclassified_as_future(self):
        clock = _Clock()
        frame = _frame(clock, 8)

        class DelayedImageClient(_ImageClient):
            def get_head_rgbd_frame(self):
                clock.advance(0.001)
                frame.received_monotonic_ns = clock()
                return super().get_head_rgbd_frame()

        capture = PreferAtomicHeadRgbdCapture(
            DelayedImageClient([frame]),
            _head_config(),
            decode_atomic_frame=_decode,
            logger=_Logger(),
            monotonic_ns=clock,
        )

        self.assertEqual(capture.read().pairing["mode"], "atomic")

    def test_missing_capability_falls_back_and_warns_only_once(self):
        clock = _Clock()
        client = _ImageClient([])
        logger = _Logger()
        config = _head_config()
        del config["rgbd_zmq_port"]
        capture = PreferAtomicHeadRgbdCapture(
            client,
            config,
            decode_atomic_frame=None,
            logger=logger,
            monotonic_ns=clock,
        )

        first = capture.read()
        second = capture.read()

        self.assertEqual(first.pairing["mode"], "legacy")
        self.assertFalse(first.pairing["paired"])
        self.assertEqual(
            first.pairing["fallback_reason"],
            "atomic_port_not_configured",
        )
        self.assertEqual(second.pairing["mode"], "legacy")
        self.assertEqual(len(logger.warnings), 1)
        self.assertEqual(client.atomic_calls, 0)
        self.assertEqual(client.legacy_color_calls, 2)

    def test_atomic_exception_never_escapes_and_repeated_failure_is_quiet(self):
        clock = _Clock()
        client = _ImageClient([RuntimeError("transport down"), RuntimeError("still down")])
        logger = _Logger()
        capture = PreferAtomicHeadRgbdCapture(
            client,
            _head_config(),
            decode_atomic_frame=_decode,
            logger=logger,
            monotonic_ns=clock,
            recovery_probe_interval_s=1.0,
        )

        first = capture.read()
        second = capture.read()
        clock.advance(1.0)
        third = capture.read()

        self.assertEqual(first.pairing["fallback_reason"], "atomic_receive_error")
        self.assertEqual(second.pairing["mode"], "legacy")
        self.assertEqual(third.pairing["mode"], "legacy")
        self.assertEqual(len(logger.warnings), 1)
        self.assertEqual(client.atomic_calls, 2)
        self.assertEqual(client.legacy_color_calls, 3)

    def test_atomic_decode_error_falls_back_without_escaping(self):
        clock = _Clock()
        client = _ImageClient([_frame(clock, 1)])
        logger = _Logger()

        def reject_frame(_frame):
            raise ValueError("bad PNG")

        capture = PreferAtomicHeadRgbdCapture(
            client,
            _head_config(),
            decode_atomic_frame=reject_frame,
            logger=logger,
            monotonic_ns=clock,
        )

        sample = capture.read()

        self.assertEqual(sample.pairing["mode"], "legacy")
        self.assertEqual(sample.pairing["fallback_reason"], "atomic_decode_error")
        self.assertEqual(len(logger.warnings), 1)
        self.assertEqual(client.legacy_color_calls, 1)

    def test_duplicate_and_no_frame_hold_sample_then_stall_falls_back(self):
        clock = _Clock()
        frame = _frame(clock, 3)
        client = _ImageClient([frame, frame, None, frame])
        logger = _Logger()
        capture = PreferAtomicHeadRgbdCapture(
            client,
            _head_config(),
            decode_atomic_frame=_decode,
            logger=logger,
            monotonic_ns=clock,
            stall_timeout_s=0.15,
        )

        self.assertEqual(capture.read().pairing["mode"], "atomic")
        clock.advance(0.03)
        duplicate = capture.read()
        self.assertTrue(duplicate.pairing["sample_held"])
        self.assertEqual(duplicate.pairing["sample_hold_count"], 1)
        self.assertEqual(duplicate.pairing["capture_sequence"], 3)
        clock.advance(0.03)
        no_frame = capture.read()
        self.assertTrue(no_frame.pairing["sample_held"])
        self.assertEqual(no_frame.pairing["sample_hold_count"], 2)
        self.assertEqual(no_frame.pairing["capture_sequence"], 3)
        self.assertEqual(client.legacy_color_calls, 0)
        clock.advance(0.16)
        fallback = capture.read()

        self.assertEqual(fallback.pairing["mode"], "legacy")
        self.assertEqual(fallback.pairing["fallback_reason"], "atomic_stale_frame")
        self.assertEqual(len(logger.warnings), 1)

        clock.advance(10.0)
        client.atomic_values.append(_frame(clock, 4))
        still_legacy = capture.read()
        self.assertEqual(still_legacy.pairing["mode"], "legacy")
        self.assertEqual(client.atomic_calls, 4)
        self.assertEqual(len(logger.warnings), 1)

    def test_recovers_automatically_and_sequence_regression_does_not_raise(self):
        clock = _Clock()
        first = _frame(clock, 10)
        client = _ImageClient([None, first])
        logger = _Logger()
        capture = PreferAtomicHeadRgbdCapture(
            client,
            _head_config(),
            decode_atomic_frame=_decode,
            logger=logger,
            monotonic_ns=clock,
            recovery_probe_interval_s=1.0,
        )

        self.assertEqual(capture.read().pairing["mode"], "legacy")
        clock.advance(0.5)
        self.assertEqual(capture.read().pairing["mode"], "legacy")
        clock.advance(0.5)
        first.received_monotonic_ns = clock()
        self.assertEqual(capture.read().pairing["capture_sequence"], 10)

        clock.advance(0.03)
        reset = _frame(clock, 2)
        client.atomic_values.append(reset)
        regressed_sample = capture.read()

        self.assertTrue(regressed_sample.pairing["capture_sequence_regressed"])
        self.assertEqual(regressed_sample.pairing["mode"], "atomic")
        self.assertEqual(len(logger.warnings), 1)
        self.assertEqual(len(logger.infos), 1)

    def test_failure_after_atomic_success_makes_legacy_fallback_sticky(self):
        clock = _Clock()
        first = _frame(clock, 1)
        client = _ImageClient([first, RuntimeError("stream failed")])
        logger = _Logger()
        capture = PreferAtomicHeadRgbdCapture(
            client,
            _head_config(),
            decode_atomic_frame=_decode,
            logger=logger,
            monotonic_ns=clock,
            recovery_probe_interval_s=0.1,
        )

        self.assertEqual(capture.read().pairing["mode"], "atomic")
        clock.advance(0.03)
        self.assertEqual(capture.read().pairing["mode"], "legacy")

        clock.advance(10.0)
        client.atomic_values.append(_frame(clock, 2))
        self.assertEqual(capture.read().pairing["mode"], "legacy")

        self.assertEqual(client.atomic_calls, 2)
        self.assertEqual(len(logger.infos), 1)
        self.assertEqual(len(logger.warnings), 1)


if __name__ == "__main__":
    unittest.main()
