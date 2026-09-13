"""Hardware-free regression coverage for Inspire FTP tactile capture."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from teleop.robot_control import inspire_tactile
from teleop.robot_control.inspire_tactile import (
    TACTILE_CELLS_PER_HAND,
    TACTILE_PAD_LENGTHS,
    TACTILE_PADS,
    InspireTactileReader,
)
from teleop.utils.episode_writer import EpisodeWriter


EXPECTED_PAD_LENGTHS = {
    "fingerone_tip_touch": 9,
    "fingerone_top_touch": 96,
    "fingerone_palm_touch": 80,
    "fingertwo_tip_touch": 9,
    "fingertwo_top_touch": 96,
    "fingertwo_palm_touch": 80,
    "fingerthree_tip_touch": 9,
    "fingerthree_top_touch": 96,
    "fingerthree_palm_touch": 80,
    "fingerfour_tip_touch": 9,
    "fingerfour_top_touch": 96,
    "fingerfour_palm_touch": 80,
    "fingerfive_tip_touch": 9,
    "fingerfive_top_touch": 96,
    "fingerfive_middle_touch": 9,
    "fingerfive_palm_touch": 96,
    "palm_touch": 112,
}


class _ManualClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def set_seconds(self, seconds):
        self.now_ns = int(round(seconds * 1e9))


class _FakeChannelSubscriber:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.callback = None
        self.queue_depth = None
        self.close_count = 0
        self.__class__.instances.append(self)

    def Init(self, callback, queue_depth):
        self.callback = callback
        self.queue_depth = queue_depth

    def Close(self):
        self.close_count += 1

    def deliver(self, message):
        self.callback(message)


def _touch_message(start):
    values = {}
    next_value = start
    for pad, length in EXPECTED_PAD_LENGTHS.items():
        values[pad] = list(range(next_value, next_value + length))
        next_value += length
    return SimpleNamespace(**values)


def _new_reader(stale_after_s=0.25):
    _FakeChannelSubscriber.instances.clear()
    clock = _ManualClock()
    message_type = object()
    reader = InspireTactileReader(
        subscriber_factory=_FakeChannelSubscriber,
        touch_message_type=message_type,
        clock_ns=clock,
        stale_after_s=stale_after_s,
    )
    return reader, tuple(_FakeChannelSubscriber.instances), clock, message_type


class TestInspireTactileReader(unittest.TestCase):
    def setUp(self):
        self.reader, subscribers, self.clock, self.message_type = _new_reader()
        self.left_subscriber, self.right_subscriber = subscribers
        self.addCleanup(self.reader.close)

    def _deliver_pair(self):
        self.left_subscriber.deliver(_touch_message(0))
        self.right_subscriber.deliver(_touch_message(2000))

    def test_subscribes_to_the_two_touch_topics(self):
        self.assertEqual(
            [self.left_subscriber.topic, self.right_subscriber.topic],
            [
                inspire_tactile.kTopicInspireFTPLeftTouch,
                inspire_tactile.kTopicInspireFTPRightTouch,
            ],
        )
        self.assertTrue(
            all(
                subscriber.message_type is self.message_type
                for subscriber in (self.left_subscriber, self.right_subscriber)
            )
        )
        self.assertEqual(
            [self.left_subscriber.queue_depth, self.right_subscriber.queue_depth],
            [10, 10],
        )

    def test_no_data_returns_none_and_warns_once(self):
        with mock.patch.object(inspire_tactile.logger_mp, "warning") as warning:
            self.assertIsNone(self.reader.get_tactiles())
            self.assertIsNone(self.reader.get_tactiles())

        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], "left, right")

        with mock.patch.object(inspire_tactile.logger_mp, "warning") as warning:
            self.assertFalse(self.reader.wait_for_data(timeout=0.0))
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[-2:], (0, 0))

    def test_wait_for_data_requires_both_valid_hands(self):
        self.left_subscriber.deliver(_touch_message(0))
        with mock.patch.object(inspire_tactile.logger_mp, "warning") as warning:
            self.assertFalse(self.reader.wait_for_data(timeout=0.0))
        self.assertEqual(warning.call_args.args[-2:], (1, 0))

        self.right_subscriber.deliver(_touch_message(2000))
        self.assertTrue(self.reader.wait_for_data(timeout=0.0))
        self.assertEqual(self.reader.received(), {"left": 1, "right": 1})

    def test_wait_for_data_rejects_a_staggered_stale_startup_pair(self):
        self.left_subscriber.deliver(_touch_message(0))
        self.clock.set_seconds(0.251)
        self.right_subscriber.deliver(_touch_message(2000))

        with mock.patch.object(inspire_tactile.logger_mp, "warning"):
            self.assertFalse(self.reader.wait_for_data(timeout=0.0))

        self.left_subscriber.deliver(_touch_message(4000))
        self.assertTrue(self.reader.wait_for_data(timeout=0.0))

    def test_rejects_a_pad_with_the_wrong_cell_count(self):
        malformed = _touch_message(0)
        malformed.fingerone_tip_touch.pop()

        with mock.patch.object(inspire_tactile.logger_mp, "warning") as warning:
            self.left_subscriber.deliver(malformed)

        warning.assert_called_once()
        self.assertIn("expected 9", str(warning.call_args.args[2]))
        self.assertEqual(self.reader.received(), {"left": 0, "right": 0})
        with mock.patch.object(inspire_tactile.logger_mp, "warning"):
            self.assertIsNone(self.reader.get_tactiles())

        self._deliver_pair()
        with mock.patch.object(inspire_tactile.logger_mp, "info"):
            self.assertIsNotNone(self.reader.get_tactiles())

    def test_20hz_sample_is_held_for_30hz_frames_without_aliasing(self):
        self._deliver_pair()

        first_record_frame = self.reader.get_tactiles()
        self.clock.set_seconds(1 / 30)
        second_record_frame = self.reader.get_tactiles()

        self.assertEqual(first_record_frame, second_record_frame)
        self.assertEqual(
            first_record_frame["left_ee"]["fingerone_tip_touch"][0],
            0,
        )
        self.assertEqual(
            first_record_frame["right_ee"]["fingerone_tip_touch"][0],
            2000,
        )
        self.assertIsNot(
            first_record_frame["left_ee"]["fingerone_tip_touch"],
            second_record_frame["left_ee"]["fingerone_tip_touch"],
        )
        self.assertEqual(self.reader.received(), {"left": 1, "right": 1})

        first_record_frame["left_ee"]["fingerone_tip_touch"][0] = -1
        third_record_frame = self.reader.get_tactiles()
        self.assertEqual(third_record_frame, second_record_frame)

        self.left_subscriber.deliver(_touch_message(4000))
        next_record_frame = self.reader.get_tactiles()
        self.assertNotEqual(next_record_frame["left_ee"], second_record_frame["left_ee"])
        self.assertEqual(next_record_frame["right_ee"], second_record_frame["right_ee"])
        self.assertEqual(self.reader.received(), {"left": 2, "right": 1})

    def test_stale_side_returns_none_until_both_sides_recover(self):
        self._deliver_pair()
        self.assertIsNotNone(self.reader.get_tactiles())

        with (
            mock.patch.object(inspire_tactile.logger_mp, "warning") as warning,
            mock.patch.object(inspire_tactile.logger_mp, "info") as info,
        ):
            self.clock.set_seconds(0.251)
            self.assertIsNone(self.reader.get_tactiles())
            self.assertIsNone(self.reader.get_tactiles())
            warning.assert_called_once()
            self.assertEqual(warning.call_args.args[1], "left, right")

            self.left_subscriber.deliver(_touch_message(4000))
            self.assertIsNone(self.reader.get_tactiles())
            warning.assert_called_once()

            self.right_subscriber.deliver(_touch_message(6000))
            recovered = self.reader.get_tactiles()

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["left_ee"]["fingerone_tip_touch"][0], 4000)
        self.assertEqual(recovered["right_ee"]["fingerone_tip_touch"][0], 6000)
        info.assert_called_once_with(
            "[InspireTactileReader] both tactile streams recovered"
        )

    def test_each_complete_hand_frame_contains_1062_json_cells(self):
        self._deliver_pair()
        tactile = self.reader.get_tactiles()

        self.assertEqual(TACTILE_PAD_LENGTHS, EXPECTED_PAD_LENGTHS)
        self.assertEqual(TACTILE_PADS, tuple(EXPECTED_PAD_LENGTHS))
        self.assertEqual(TACTILE_CELLS_PER_HAND, 1062)
        for side in ("left_ee", "right_ee"):
            self.assertEqual(
                {pad: len(values) for pad, values in tactile[side].items()},
                EXPECTED_PAD_LENGTHS,
            )
            self.assertEqual(sum(map(len, tactile[side].values())), 1062)
            self.assertTrue(
                all(isinstance(values, list) for values in tactile[side].values())
            )
        json.dumps(tactile)

    def test_close_is_idempotent_and_ignores_late_callbacks(self):
        self.reader.close()
        self.reader.close()

        self.assertEqual(self.left_subscriber.close_count, 1)
        self.assertEqual(self.right_subscriber.close_count, 1)
        self.left_subscriber.deliver(_touch_message(0))
        self.right_subscriber.deliver(_touch_message(2000))
        self.assertEqual(self.reader.received(), {"left": 0, "right": 0})


class TestTactileRecordingContract(unittest.TestCase):
    @staticmethod
    def _teleop_tree():
        source_path = Path(__file__).parents[1] / "teleop_hand_and_arm.py"
        return ast.parse(source_path.read_text(encoding="utf-8"))

    @staticmethod
    def _is_ftp_guard(node):
        if not isinstance(node, ast.If):
            return False
        return any(
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and isinstance(test.left.value, ast.Name)
            and test.left.value.id == "args"
            and test.left.attr == "ee"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "inspire_ftp"
            for test in ast.walk(node.test)
        )

    def test_teleop_initializes_and_describes_tactile_capture_for_ftp(self):
        tree = self._teleop_tree()
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        def is_under_ftp_guard(node):
            while node in parents:
                node = parents[node]
                if self._is_ftp_guard(node):
                    return True
            return False

        def guard_text(node):
            guards = []
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.If):
                    guards.append(ast.unparse(node.test))
            return " ".join(guards)

        constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "InspireTactileReader"
        ]
        waits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wait_for_data"
        ]
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_tactiles"
        ]
        closes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "close"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tactile_reader"
        ]
        assignments = {
            ast.unparse(target): node.value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        }

        self.assertTrue(constructors)
        self.assertTrue(all(is_under_ftp_guard(node) for node in constructors))
        for constructor in constructors:
            conditions = guard_text(constructor)
            self.assertIn("args.record", conditions)
            self.assertIn("not args.sim", conditions)
        self.assertTrue(any(is_under_ftp_guard(node) for node in waits))
        self.assertTrue(reads)
        self.assertTrue(closes)
        for side in ("left_ee", "right_ee"):
            target = f"recorder.info['tactile_names']['{side}']"
            self.assertIn(target, assignments)
            self.assertTrue(
                any(
                    isinstance(node, ast.Name) and node.id == "TACTILE_PADS"
                    for node in ast.walk(assignments[target])
                )
            )

    def test_every_recorder_call_passes_tactiles(self):
        tree = self._teleop_tree()
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "recorder"
            and node.func.attr == "add_item"
        ]

        self.assertGreater(len(calls), 0)
        for call in calls:
            self.assertIn("tactiles", {keyword.arg for keyword in call.keywords})

    def test_writer_preserves_names_repeats_and_stale_nulls(self):
        reader, (left_subscriber, right_subscriber), clock, _ = _new_reader()
        left_subscriber.deliver(_touch_message(0))
        right_subscriber.deliver(_touch_message(2000))

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                task_dir = Path(temp_dir) / "episodes"
                writer = EpisodeWriter(task_dir=str(task_dir), rerun_log=False)
                writer.info["tactile_names"]["left_ee"] = list(TACTILE_PADS)
                writer.info["tactile_names"]["right_ee"] = list(TACTILE_PADS)
                self.assertTrue(writer.create_episode())
                writer.add_item(colors={}, tactiles=reader.get_tactiles())
                clock.set_seconds(1 / 30)
                writer.add_item(colors={}, tactiles=reader.get_tactiles())
                clock.set_seconds(0.251)
                with mock.patch.object(inspire_tactile.logger_mp, "warning"):
                    writer.add_item(colors={}, tactiles=reader.get_tactiles())
                writer.close()
                saved = json.loads(
                    (task_dir / "episode_0000" / "data.json").read_text(
                        encoding="utf-8"
                    )
                )
        finally:
            reader.close()

        self.assertEqual(
            saved["info"]["tactile_names"]["left_ee"],
            list(TACTILE_PADS),
        )
        self.assertEqual(
            saved["info"]["tactile_names"]["right_ee"],
            list(TACTILE_PADS),
        )
        self.assertEqual(saved["data"][0]["tactiles"], saved["data"][1]["tactiles"])
        self.assertIsNone(saved["data"][2]["tactiles"])


if __name__ == "__main__":
    unittest.main()
