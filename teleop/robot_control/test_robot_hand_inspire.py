"""Hardware-free regression coverage for Inspire DFX and FTP support."""

import json
import threading
import time
import unittest

import numpy as np

from teleop.robot_control.robot_hand_inspire import (
    INSPIRE_LEFT_JOINT_NAMES,
    INSPIRE_RIGHT_JOINT_NAMES,
    UINT32_MAX,
    Inspire_Controller_DFX,
    InspireDFXLostCounterTracker,
    Inspire_Controller_FTP,
    _normalize_inspire_targets,
    get_inspire_end_effector_info,
    inspire_xr_hand_data_is_ready,
    kTopicInspireDFXState,
    kTopicInspireFTPLeftState,
    kTopicInspireFTPRightState,
)
from teleop.utils.subscriber_drop_tracker import SubscriberDropTracker


class _StopReader(Exception):
    pass


class _SharedArray:
    def __init__(self, values, on_write=None):
        self._values = list(values)
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._on_write = on_write

    def get_lock(self):
        return self._lock

    def __getitem__(self, index):
        with self._lock:
            return self._values[index]

    def __setitem__(self, index, values):
        with self._changed:
            self._values[index] = values
            self._changed.notify_all()
        if self._on_write is not None:
            self._on_write()

    def snapshot(self):
        with self._lock:
            return list(self._values)

    def wait_for(self, expected, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._changed:
            while not np.allclose(self._values, expected):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._changed.wait(remaining):
                    return False
            return True


class _ReadyValue:
    def __init__(self, value):
        self.value = value
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


class _ManualClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def set_seconds(self, seconds):
        self.now_ns = int(round(seconds * 1e9))


class _DFXMessage:
    def __init__(self, positions, lost=None):
        if lost is None:
            lost = [0] * len(positions)
        if len(lost) != len(positions):
            raise ValueError("test positions and counters must have the same length")
        self.states = [
            type("MotorState", (), {"q": q, "lost": counter})()
            for q, counter in zip(positions, lost, strict=True)
        ]


class _FTPMessage:
    def __init__(self, positions):
        self.angle_act = list(positions)


class _OneShotSubscriber:
    def __init__(self, message):
        self._message = message
        self._read = False

    def Read(self):
        if not self._read:
            self._read = True
            return self._message
        raise _StopReader


class _SequenceSubscriber:
    def __init__(self, messages):
        self._messages = iter(messages)

    def Read(self):
        try:
            return next(self._messages)
        except StopIteration as error:
            raise _StopReader from error


class _BlockingSubscriber:
    def __init__(self, blocked, release, first, second):
        self._blocked = blocked
        self._release = release
        self._messages = iter((first, second))
        self._calls = 0

    def Read(self):
        self._calls += 1
        if self._calls == 2:
            self._blocked.set()
            if not self._release.wait(2.0):
                raise AssertionError("test did not release the blocked reader")
        if self._calls <= 2:
            return next(self._messages)
        raise _StopReader


class _AdvancingSubscriber:
    def __init__(self, other_blocked, finish, first, second):
        self._other_blocked = other_blocked
        self._finish = finish
        self._messages = iter((first, second))
        self._calls = 0

    def Read(self):
        self._calls += 1
        if self._calls == 2 and not self._other_blocked.wait(2.0):
            raise AssertionError("other reader did not enter its simulated stall")
        if self._calls <= 2:
            return next(self._messages)
        if not self._finish.wait(2.0):
            raise AssertionError("test did not stop the advancing reader")
        raise _StopReader


def _run_reader(target, errors):
    try:
        target()
    except _StopReader:
        pass
    except Exception as error:
        errors.append(error)


def _make_ftp_reader_controller():
    controller = object.__new__(Inspire_Controller_FTP)
    controller.initial_state_received = {
        "left": threading.Event(),
        "right": threading.Event(),
    }
    controller.subscriber_drop_tracker = SubscriberDropTracker(
        {
            "left": kTopicInspireFTPLeftState,
            "right": kTopicInspireFTPRightState,
        },
        0.075,
        aggregate_count_key="total_side_gap_count",
    )
    return controller


def _make_dfx_reader_controller(left_state=None, right_state=None, clock=None):
    controller = object.__new__(Inspire_Controller_DFX)
    controller.left_hand_state_array = _SharedArray(
        [0.0] * 6 if left_state is None else left_state
    )
    controller.right_hand_state_array = _SharedArray(
        [0.0] * 6 if right_state is None else right_state
    )
    controller.hand_state_lock = threading.Lock()
    controller.initial_state_received = threading.Event()
    controller._initial_side_state_accepted = {"left": False, "right": False}
    controller.subscriber_drop_tracker = SubscriberDropTracker(
        {"combined": kTopicInspireDFXState},
        0.075,
        clock_ns=clock,
    )
    controller.lost_counter_tracker = InspireDFXLostCounterTracker(clock_ns=clock)
    return controller


class TestInspireStateReaders(unittest.TestCase):
    def test_dfx_maps_combined_message_and_accepts_all_zero_valid_state(self):
        controller = _make_dfx_reader_controller([1.0] * 6, [1.0] * 6)
        controller.HandState_subscriber = _SequenceSubscriber(
            [_DFXMessage([0.0] * 12), _DFXMessage([0.0] * 12)]
        )

        controller.subscriber_drop_tracker.begin_episode_window()
        controller.lost_counter_tracker.begin_episode_window()
        with self.assertRaises(_StopReader):
            controller._subscribe_hand_state()
        summary = controller.subscriber_drop_tracker.finish_episode_window()
        lost_summary = controller.lost_counter_tracker.finish_episode_window()

        self.assertTrue(controller.initial_state_received.is_set())
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.0] * 6)
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.0] * 6)
        self.assertEqual(summary["combined"]["sample_count"], 2)
        self.assertEqual(summary["combined"]["topic"], kTopicInspireDFXState)
        self.assertNotIn("left", summary)
        self.assertNotIn("right", summary)
        self.assertEqual(lost_summary["left"]["baseline_count"], 1)
        self.assertEqual(lost_summary["left"]["accepted_count"], 1)
        self.assertEqual(lost_summary["right"]["baseline_count"], 1)
        self.assertEqual(lost_summary["right"]["accepted_count"], 1)

    def test_dfx_combined_id_order_is_right_then_left_on_wire(self):
        positions = np.arange(12, dtype=np.float64) / 11.0
        left, right = Inspire_Controller_DFX._decode_combined_state(
            _DFXMessage(positions)
        )
        np.testing.assert_array_equal(left, positions[6:12])
        np.testing.assert_array_equal(right, positions[0:6])

    def test_dfx_rejects_non_exact_combined_state_length(self):
        with self.assertRaisesRegex(ValueError, "expected 12 combined motor states"):
            Inspire_Controller_DFX._decode_combined_state(
                _DFXMessage([0.5] * 13)
            )

    def test_dfx_combined_state_update_is_atomic(self):
        left_written = threading.Event()
        release_left_write = threading.Event()
        consumer_finished = threading.Event()
        consumer_snapshot = []
        errors = []

        def pause_after_left_write():
            left_written.set()
            if not release_left_write.wait(2.0):
                raise AssertionError("test did not release the DFX state writer")

        controller = _make_dfx_reader_controller([0.2] * 6, [0.8] * 6)
        controller.left_hand_state_array._on_write = pause_after_left_write
        controller.HandState_subscriber = _SequenceSubscriber(
            [
                _DFXMessage([0.1] * 6 + [0.9] * 6),
                _DFXMessage([0.1] * 6 + [0.9] * 6),
            ]
        )

        def read_both_sides():
            if not left_written.wait(2.0):
                errors.append(AssertionError("left side was never written"))
                return
            with controller.hand_state_lock:
                consumer_snapshot.extend(
                    controller.left_hand_state_array.snapshot()
                    + controller.right_hand_state_array.snapshot()
                )
            consumer_finished.set()

        subscriber_thread = threading.Thread(
            target=_run_reader,
            args=(controller._subscribe_hand_state, errors),
        )
        consumer_thread = threading.Thread(target=read_both_sides)
        subscriber_thread.start()
        consumer_thread.start()
        try:
            self.assertTrue(left_written.wait(2.0))
            self.assertFalse(
                consumer_finished.wait(0.05),
                "consumer observed state between the left and right updates",
            )
            release_left_write.set()
        finally:
            release_left_write.set()
            subscriber_thread.join(2.0)
            consumer_thread.join(2.0)

        self.assertFalse(subscriber_thread.is_alive())
        self.assertFalse(consumer_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(consumer_snapshot, [0.9] * 6 + [0.1] * 6)

    def test_dfx_combined_gap_is_counted_once(self):
        clock = _ManualClock()
        tracker = SubscriberDropTracker(
            {"combined": kTopicInspireDFXState},
            0.075,
            clock_ns=clock,
        )
        tracker.observe("combined")
        tracker.begin_episode_window()
        clock.set_seconds(1.0)
        tracker.observe("combined")
        summary = tracker.finish_episode_window()

        self.assertEqual(summary["combined"]["gap_count"], 1)
        self.assertEqual(summary["combined"]["recovered_gap_count"], 1)
        self.assertEqual(summary["total_stream_gap_count"], 1)

    def test_dfx_out_of_range_message_does_not_update_or_set_ready(self):
        controller = _make_dfx_reader_controller([0.25] * 6, [0.75] * 6)
        positions = [0.5] * 12
        positions[8] = 1.01
        controller.HandState_subscriber = _OneShotSubscriber(_DFXMessage(positions))

        controller.subscriber_drop_tracker.begin_episode_window()
        controller.lost_counter_tracker.begin_episode_window()
        with self.assertRaises(_StopReader):
            controller._subscribe_hand_state()
        summary = controller.subscriber_drop_tracker.finish_episode_window()
        lost_summary = controller.lost_counter_tracker.finish_episode_window()

        self.assertFalse(controller.initial_state_received.is_set())
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.25] * 6)
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.75] * 6)
        self.assertEqual(summary["combined"]["sample_count"], 0)
        self.assertFalse(summary["combined"]["has_received_valid_sample"])
        self.assertEqual(lost_summary["left"]["sample_count"], 0)
        self.assertEqual(lost_summary["right"]["sample_count"], 0)
        self.assertEqual(lost_summary["malformed_message_count"], 1)
        self.assertTrue(lost_summary["any_anomaly"])

    def test_dfx_lost_increment_holds_only_failed_side_then_recovers(self):
        clock = _ManualClock()
        controller = _make_dfx_reader_controller(clock=clock)

        # The first coherent counters are baseline-only. The second unchanged
        # sample proves both sides are readable and accepts all-zero as valid q.
        controller._process_combined_state_message(
            _DFXMessage([0.4] * 6 + [0.0] * 6),
            received_ns=clock(),
        )
        self.assertFalse(controller.initial_state_received.is_set())
        clock.set_seconds(0.01)
        controller._process_combined_state_message(
            _DFXMessage([0.4] * 6 + [0.0] * 6),
            received_ns=clock(),
        )
        self.assertTrue(controller.initial_state_received.is_set())
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.0] * 6)
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.4] * 6)

        controller.lost_counter_tracker.begin_episode_window()
        clock.set_seconds(0.02)
        controller._process_combined_state_message(
            _DFXMessage(
                [0.9] * 6 + [0.2] * 6,
                lost=[1] * 6 + [0] * 6,
            ),
            received_ns=clock(),
        )
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.2] * 6)
        self.assertEqual(
            controller.right_hand_state_array.snapshot(),
            [0.4] * 6,
            "right q must remain at the last accepted sample while lost advances",
        )

        clock.set_seconds(0.03)
        controller._process_combined_state_message(
            _DFXMessage(
                [0.7] * 6 + [0.3] * 6,
                lost=[1] * 6 + [0] * 6,
            ),
            received_ns=clock(),
        )
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.3] * 6)
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.7] * 6)

        clock.set_seconds(0.04)
        summary = controller.lost_counter_tracker.finish_episode_window()
        self.assertEqual(summary["right"]["drop_event_count"], 1)
        self.assertEqual(summary["right"]["lost_increment_count"], 1)
        self.assertEqual(summary["right"]["held_sample_count"], 1)
        self.assertEqual(summary["right"]["accepted_count"], 1)
        self.assertEqual(summary["left"]["drop_event_count"], 0)
        self.assertEqual(summary["left"]["accepted_count"], 2)
        self.assertEqual(summary["total_drop_event_count"], 1)
        self.assertEqual(summary["total_lost_increment_count"], 1)
        self.assertEqual(summary["total_held_sample_count"], 1)
        self.assertTrue(summary["any_state_held"])
        self.assertTrue(summary["any_anomaly"])
        json.dumps(summary)

    def test_dfx_counter_regression_requires_new_baseline_and_clean_sample(self):
        clock = _ManualClock()
        controller = _make_dfx_reader_controller(clock=clock)
        baseline_message = _DFXMessage(
            [0.4] * 6 + [0.2] * 6,
            lost=[5] * 12,
        )
        controller._process_combined_state_message(
            baseline_message,
            received_ns=clock(),
        )
        clock.set_seconds(0.01)
        controller._process_combined_state_message(
            baseline_message,
            received_ns=clock(),
        )

        controller.lost_counter_tracker.begin_episode_window()
        regression = _DFXMessage(
            [0.9] * 6 + [0.3] * 6,
            lost=[0] * 6 + [5] * 6,
        )
        clock.set_seconds(0.02)
        controller._process_combined_state_message(
            regression,
            received_ns=clock(),
        )
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.4] * 6)
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.3] * 6)

        new_baseline = _DFXMessage(
            [0.8] * 6 + [0.4] * 6,
            lost=[0] * 6 + [5] * 6,
        )
        clock.set_seconds(0.03)
        controller._process_combined_state_message(
            new_baseline,
            received_ns=clock(),
        )
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.4] * 6)
        clock.set_seconds(0.04)
        controller._process_combined_state_message(
            new_baseline,
            received_ns=clock(),
        )
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.8] * 6)

        clock.set_seconds(0.05)
        summary = controller.lost_counter_tracker.finish_episode_window()
        right = summary["right"]
        self.assertEqual(right["counter_regression_count"], 1)
        self.assertEqual(right["reset_count"], 1)
        self.assertEqual(right["baseline_count"], 1)
        self.assertEqual(right["held_sample_count"], 2)
        self.assertEqual(right["accepted_count"], 1)

    def test_dfx_counter_validation_and_divergence_are_fail_closed(self):
        tracker = InspireDFXLostCounterTracker()
        with self.assertRaisesRegex(ValueError, "expected 6 lost counters"):
            tracker.observe_combined([0] * 5, [0] * 6)
        with self.assertRaisesRegex(ValueError, "must not be boolean"):
            tracker.observe_combined([False] * 6, [0] * 6)
        with self.assertRaisesRegex(ValueError, "outside uint32"):
            tracker.observe_combined([UINT32_MAX + 1] * 6, [0] * 6)

        clock = _ManualClock()
        controller = _make_dfx_reader_controller(
            [0.2] * 6,
            [0.4] * 6,
            clock=clock,
        )
        stable = _DFXMessage([0.4] * 6 + [0.2] * 6)
        controller._process_combined_state_message(stable, received_ns=clock())
        clock.set_seconds(0.01)
        controller._process_combined_state_message(stable, received_ns=clock())
        controller.lost_counter_tracker.begin_episode_window()
        clock.set_seconds(0.02)
        controller._process_combined_state_message(
            _DFXMessage(
                [0.9] * 6 + [0.3] * 6,
                lost=[1, 1, 2, 1, 1, 1] + [0] * 6,
            ),
            received_ns=clock(),
        )
        self.assertEqual(controller.right_hand_state_array.snapshot(), [0.4] * 6)
        self.assertEqual(controller.left_hand_state_array.snapshot(), [0.3] * 6)
        clock.set_seconds(0.03)
        summary = controller.lost_counter_tracker.finish_episode_window()
        self.assertEqual(summary["right"]["counter_divergence_count"], 1)
        self.assertEqual(summary["right"]["reset_count"], 1)

    def test_ftp_left_stall_does_not_freeze_right(self):
        controller = _make_ftp_reader_controller()
        left_state = _SharedArray([0.0] * 6)
        right_state = _SharedArray([0.0] * 6)
        left_blocked = threading.Event()
        release_left = threading.Event()
        finish_right = threading.Event()
        errors = []

        left_thread = threading.Thread(
            target=_run_reader,
            args=(
                lambda: controller._subscribe_hand_state(
                    _BlockingSubscriber(
                        left_blocked,
                        release_left,
                        _FTPMessage([100] * 6),
                        _FTPMessage([200] * 6),
                    ),
                    left_state,
                    "left",
                ),
                errors,
            ),
        )
        right_thread = threading.Thread(
            target=_run_reader,
            args=(
                lambda: controller._subscribe_hand_state(
                    _AdvancingSubscriber(
                        left_blocked,
                        finish_right,
                        _FTPMessage([700] * 6),
                        _FTPMessage([800] * 6),
                    ),
                    right_state,
                    "right",
                ),
                errors,
            ),
        )

        left_thread.start()
        right_thread.start()
        try:
            self.assertTrue(left_blocked.wait(2.0))
            self.assertTrue(left_state.wait_for([0.1] * 6))
            self.assertTrue(right_state.wait_for([0.8] * 6))
            release_left.set()
            self.assertTrue(left_state.wait_for([0.2] * 6))
        finally:
            release_left.set()
            finish_right.set()
            left_thread.join(2.0)
            right_thread.join(2.0)

        self.assertFalse(left_thread.is_alive())
        self.assertFalse(right_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(controller.initial_state_received["left"].is_set())
        self.assertTrue(controller.initial_state_received["right"].is_set())

    def test_ftp_out_of_range_message_does_not_update_or_set_ready(self):
        controller = _make_ftp_reader_controller()
        state = _SharedArray([0.4] * 6)
        subscriber = _OneShotSubscriber(_FTPMessage([0, 200, 400, 600, 800, 1001]))

        controller.subscriber_drop_tracker.begin_episode_window()
        with self.assertRaises(_StopReader):
            controller._subscribe_hand_state(subscriber, state, "left")
        summary = controller.subscriber_drop_tracker.finish_episode_window()

        self.assertEqual(state.snapshot(), [0.4] * 6)
        self.assertFalse(controller.initial_state_received["left"].is_set())
        self.assertEqual(summary["left"]["sample_count"], 0)
        self.assertFalse(summary["left"]["has_received_valid_sample"])


class TestInspireCommandsAndContract(unittest.TestCase):
    @staticmethod
    def _publisher():
        class Publisher:
            def __init__(self):
                self.messages = []

            def Write(self, message):
                self.messages.append(message)

        return Publisher()

    def test_dfx_command_maps_left_first_model_to_right_first_wire(self):
        controller = object.__new__(Inspire_Controller_DFX)
        controller.hand_msg = type("Command", (), {})()
        controller.hand_msg.cmds = [
            type("MotorCommand", (), {"q": -1.0})() for _ in range(12)
        ]
        controller.HandCmb_publisher = self._publisher()

        left = np.linspace(0.1, 0.6, 6)
        right = np.linspace(0.9, 0.4, 6)
        controller.ctrl_dual_hand(left, right)

        wire_values = [command.q for command in controller.hand_msg.cmds]
        np.testing.assert_allclose(wire_values, np.concatenate((right, left)))
        self.assertEqual(len(controller.HandCmb_publisher.messages), 1)

        previous_wire_values = list(wire_values)
        with self.assertRaisesRegex(ValueError, "outside \\[0, 1\\]"):
            controller.ctrl_dual_hand(left, [0.5] * 5 + [1.01])
        self.assertEqual(len(controller.HandCmb_publisher.messages), 1)
        self.assertEqual(
            [command.q for command in controller.hand_msg.cmds],
            previous_wire_values,
        )

    def test_ftp_command_factory_is_instance_owned_and_writes_both_sides(self):
        class Message:
            angle_set = None
            mode = None

        class Publisher:
            def __init__(self):
                self.messages = []

            def Write(self, message):
                self.messages.append(message)

        controller = object.__new__(Inspire_Controller_FTP)
        controller._new_ftp_control_message = Message
        controller.LeftHandCmd_publisher = Publisher()
        controller.RightHandCmd_publisher = Publisher()
        controller._debug_count = 50
        controller._send_hand_command([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11])

        left_message = controller.LeftHandCmd_publisher.messages[0]
        right_message = controller.RightHandCmd_publisher.messages[0]
        self.assertEqual(left_message.angle_set, [0, 1, 2, 3, 4, 5])
        self.assertEqual(right_message.angle_set, [6, 7, 8, 9, 10, 11])
        self.assertEqual(left_message.mode, 1)
        self.assertEqual(right_message.mode, 1)

        with self.assertRaisesRegex(ValueError, "right command is outside"):
            controller._send_hand_command(
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 1001],
            )
        self.assertEqual(len(controller.LeftHandCmd_publisher.messages), 1)
        self.assertEqual(len(controller.RightHandCmd_publisher.messages), 1)

    def test_ftp_control_does_not_publish_before_xr_ready(self):
        for ready_value in (_ReadyValue(False), None):
            with self.subTest(has_explicit_ready_flag=ready_value is not None):
                controller = object.__new__(Inspire_Controller_FTP)
                controller.fps = 100_000.0
                controller._send_hand_command = lambda *_args: self.fail(
                    "command published before XR data was ready"
                )
                left_state = _SharedArray([0.2] * 6)
                right_state = _SharedArray([0.8] * 6)
                state_output = _SharedArray([0.0] * 12)
                action_output = _SharedArray(
                    [0.0] * 12,
                    on_write=lambda: setattr(controller, "running", False),
                )

                controller.control_process(
                    _SharedArray([0.0] * 75),
                    _SharedArray([0.0] * 75),
                    left_state,
                    right_state,
                    threading.Lock(),
                    state_output,
                    action_output,
                    ready_value,
                )

                self.assertEqual(state_output.snapshot(), [0.2] * 6 + [0.8] * 6)
                self.assertEqual(action_output.snapshot(), [0.2] * 6 + [0.8] * 6)

    def test_dfx_control_does_not_publish_before_xr_ready(self):
        controller = object.__new__(Inspire_Controller_DFX)
        controller.fps = 100_000.0
        controller.ctrl_dual_hand = lambda *_args: self.fail(
            "command published before XR data was ready"
        )
        left_state = _SharedArray([0.2] * 6)
        right_state = _SharedArray([0.8] * 6)
        state_output = _SharedArray([0.0] * 12)
        action_output = _SharedArray(
            [0.0] * 12,
            on_write=lambda: setattr(controller, "running", False),
        )

        controller.control_process(
            _SharedArray([0.0] * 75),
            _SharedArray([0.0] * 75),
            left_state,
            right_state,
            threading.Lock(),
            state_output,
            action_output,
            threading.Lock(),
            None,
        )

        self.assertEqual(state_output.snapshot(), [0.2] * 6 + [0.8] * 6)
        self.assertEqual(action_output.snapshot(), [0.2] * 6 + [0.8] * 6)

    def test_normalization_and_xr_fallback_validation(self):
        np.testing.assert_allclose(
            _normalize_inspire_targets([0.0, 0.0, 0.0, 0.0, 0.0, -0.1]),
            np.ones(6),
        )
        np.testing.assert_allclose(
            _normalize_inspire_targets([1.7, 1.7, 1.7, 1.7, 0.5, 1.3]),
            np.zeros(6),
        )
        valid_hand = np.zeros((25, 3))
        valid_hand[:, 0] = np.linspace(0.0, 0.2, 25)
        self.assertTrue(inspire_xr_hand_data_is_ready(valid_hand, valid_hand))
        self.assertFalse(
            inspire_xr_hand_data_is_ready(np.zeros((25, 3)), np.ones((25, 3)))
        )
        self.assertFalse(
            inspire_xr_hand_data_is_ready(np.ones((25, 3)), np.ones((25, 3)))
        )

    def test_end_effector_contract_is_exact_for_both_protocols(self):
        for protocol in ("dfx", "ftp"):
            info = get_inspire_end_effector_info(protocol)
            self.assertEqual(info["schema_version"], 1)
            self.assertEqual(info["type"], "inspire")
            self.assertEqual(info["protocol"], protocol)
            self.assertEqual(info["hand_dof"], 6)
            self.assertEqual(info["value_unit"], "normalized_open_fraction")
            self.assertEqual(info["value_range"], [0.0, 1.0])
            self.assertEqual(info["zero_semantics"], "fully_closed")
            self.assertEqual(info["one_semantics"], "fully_open")
            self.assertEqual(info["left_joint_names"], list(INSPIRE_LEFT_JOINT_NAMES))
            self.assertEqual(info["right_joint_names"], list(INSPIRE_RIGHT_JOINT_NAMES))
            self.assertEqual(info["canonical_order"], "left_then_right")


if __name__ == "__main__":
    unittest.main()
