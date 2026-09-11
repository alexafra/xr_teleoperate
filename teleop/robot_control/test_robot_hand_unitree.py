"""Hardware-free regression coverage for independent Dex3 state readers."""

import threading
import time
import unittest

from teleop.robot_control.robot_hand_unitree import (
    Dex3_1_Controller,
    Dex3_1_Left_JointIndex,
    Dex3_1_Right_JointIndex,
    Dex3SubscriberDropTracker,
)


class _StopReader(Exception):
    pass


class _HandMessage:
    def __init__(self, positions):
        self.motor_state = [type("MotorState", (), {"q": value})() for value in positions]


class _SharedArray:
    def __init__(self, values):
        self._values = list(values)
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    def get_lock(self):
        return self._lock

    def __getitem__(self, index):
        with self._lock:
            return self._values[index]

    def __setitem__(self, index, values):
        with self._changed:
            self._values[index] = values
            self._changed.notify_all()

    def snapshot(self):
        with self._lock:
            return list(self._values)

    def wait_for(self, expected, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._changed:
            while self._values != expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._changed.wait(remaining):
                    return False
            return True


class _LeftSubscriber:
    def __init__(self, blocked, release, initial, recovered):
        self._blocked = blocked
        self._release = release
        self._messages = iter((_HandMessage(initial), _HandMessage(recovered)))
        self._calls = 0

    def Read(self):
        self._calls += 1
        if self._calls == 2:
            self._blocked.set()
            if not self._release.wait(2.0):
                raise AssertionError("test did not release the stalled left reader")
        if self._calls <= 2:
            return next(self._messages)
        raise _StopReader


class _RightSubscriber:
    def __init__(self, left_blocked, finish, initial, advanced):
        self._left_blocked = left_blocked
        self._finish = finish
        self._messages = iter((_HandMessage(initial), _HandMessage(advanced)))
        self._calls = 0

    def Read(self):
        self._calls += 1
        if self._calls == 2 and not self._left_blocked.wait(2.0):
            raise AssertionError("left reader did not enter its simulated stall")
        if self._calls <= 2:
            return next(self._messages)
        if not self._finish.wait(2.0):
            raise AssertionError("test did not stop the right reader")
        raise _StopReader


def _run_reader(controller, subscriber, state_array, joint_indices, side, errors):
    try:
        controller._subscribe_hand_state(subscriber, state_array, joint_indices, side)
    except _StopReader:
        pass
    except Exception as error:
        errors.append(error)


def _one_control_snapshot(controller, left_state, right_state):
    left_input = _SharedArray([0.0] * 75)
    right_input = _SharedArray([0.0] * 75)
    state_output = _SharedArray([0.0] * 14)
    action_output = _SharedArray([0.0] * 14)

    controller.fps = 100_000.0

    def stop_after_output(_left_target, _right_target):
        controller.running = False

    controller.ctrl_dual_hand = stop_after_output
    controller.control_process(
        left_input,
        right_input,
        left_state,
        right_state,
        threading.Lock(),
        state_output,
        action_output,
    )
    return state_output.snapshot()


class TestDex3IndependentReaders(unittest.TestCase):
    def test_left_stall_does_not_freeze_right_and_left_recovers_in_place(self):
        left_initial = [10.0 + index for index in range(7)]
        left_recovered = [20.0 + index for index in range(7)]
        right_initial = [110.0 + index for index in range(7)]
        right_advanced = [120.0 + index for index in range(7)]

        left_state = _SharedArray([0.0] * 7)
        right_state = _SharedArray([0.0] * 7)
        left_blocked = threading.Event()
        release_left = threading.Event()
        finish_right = threading.Event()
        errors = []
        controller = object.__new__(Dex3_1_Controller)
        controller.subscriber_drop_tracker = Dex3SubscriberDropTracker()

        left_thread = threading.Thread(
            target=_run_reader,
            args=(
                controller,
                _LeftSubscriber(left_blocked, release_left, left_initial, left_recovered),
                left_state,
                Dex3_1_Left_JointIndex,
                "left",
                errors,
            ),
        )
        right_thread = threading.Thread(
            target=_run_reader,
            args=(
                controller,
                _RightSubscriber(left_blocked, finish_right, right_initial, right_advanced),
                right_state,
                Dex3_1_Right_JointIndex,
                "right",
                errors,
            ),
        )

        left_thread.start()
        right_thread.start()
        try:
            self.assertTrue(left_blocked.wait(2.0), "left reader never entered its stall")
            self.assertTrue(left_state.wait_for(left_initial))
            self.assertTrue(
                right_state.wait_for(right_advanced),
                "right state did not advance while the left Read was blocked",
            )

            self.assertEqual(left_state.snapshot(), left_initial)
            self.assertEqual(right_state.snapshot(), right_advanced)
            self.assertEqual(
                _one_control_snapshot(controller, left_state, right_state),
                left_initial + right_advanced,
                "combined state must remain left[7] followed by right[7]",
            )

            release_left.set()
            self.assertTrue(left_state.wait_for(left_recovered), "left state did not recover")
            self.assertEqual(
                right_state.snapshot(),
                right_advanced,
                "left recovery must not rewrite the cached right state",
            )
            self.assertEqual(
                _one_control_snapshot(controller, left_state, right_state),
                left_recovered + right_advanced,
            )
        finally:
            release_left.set()
            finish_right.set()
            left_thread.join(2.0)
            right_thread.join(2.0)

        self.assertFalse(left_thread.is_alive())
        self.assertFalse(right_thread.is_alive())
        self.assertEqual(errors, [])


class _ManualClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def set_seconds(self, seconds):
        self.now_ns = int(round(seconds * 1e9))


class _BlockingFinishClock:
    def __init__(self):
        self.finish_clock_entered = threading.Event()
        self.release_finish_clock = threading.Event()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            return 0
        self.finish_clock_entered.set()
        if not self.release_finish_clock.wait(2.0):
            raise AssertionError("test did not release the finish clock")
        return 1_000_000_000


class TestDex3SubscriberDropTracker(unittest.TestCase):
    def test_threshold_is_strict_and_sides_are_independent(self):
        clock = _ManualClock()
        tracker = Dex3SubscriberDropTracker(clock_ns=clock)
        tracker.observe("left")
        tracker.observe("right")
        tracker.begin_episode_window()

        clock.set_seconds(0.074)
        tracker.observe("right")
        clock.set_seconds(0.075)
        tracker.observe("left")
        clock.set_seconds(0.150000001)
        tracker.observe("left")
        clock.set_seconds(0.151)
        summary = tracker.finish_episode_window()

        self.assertEqual(summary["left"]["gap_count"], 1)
        self.assertEqual(summary["left"]["recovered_gap_count"], 1)
        self.assertFalse(summary["left"]["open_gap_at_end"])
        self.assertAlmostEqual(summary["left"]["gaps"][0]["duration_s"], 0.075000001)
        self.assertEqual(summary["right"]["gap_count"], 1)
        self.assertEqual(summary["right"]["recovered_gap_count"], 0)
        self.assertTrue(summary["right"]["open_gap_at_end"])

    def test_open_gap_at_stop_is_counted_and_marked_unrecovered(self):
        clock = _ManualClock()
        tracker = Dex3SubscriberDropTracker(clock_ns=clock)
        tracker.observe("left")
        tracker.observe("right")
        tracker.begin_episode_window()

        clock.set_seconds(1.004)
        tracker.observe("left")
        summary = tracker.finish_episode_window()

        self.assertEqual(summary["left"]["gap_count"], 1)
        self.assertTrue(summary["left"]["gaps"][0]["recovered"])
        right_summary = summary["right"]
        open_gap = right_summary["gaps"][0]
        self.assertEqual(right_summary["gap_count"], 1)
        self.assertTrue(right_summary["open_gap_at_end"])
        self.assertFalse(open_gap["recovered"])
        self.assertAlmostEqual(open_gap["duration_s"], 1.004)

    def test_gap_is_clipped_to_episode_and_windows_do_not_leak(self):
        clock = _ManualClock()
        tracker = Dex3SubscriberDropTracker(clock_ns=clock)
        tracker.observe("left")

        clock.set_seconds(0.5)
        tracker.begin_episode_window()
        clock.set_seconds(1.0)
        tracker.observe("left")
        first = tracker.finish_episode_window()
        self.assertAlmostEqual(first["left"]["gaps"][0]["start_offset_s"], 0.0)
        self.assertAlmostEqual(first["left"]["gaps"][0]["duration_s"], 0.5)

        clock.set_seconds(2.0)
        tracker.observe("left")
        tracker.begin_episode_window()
        clock.set_seconds(2.01)
        tracker.observe("left")
        second = tracker.finish_episode_window()
        self.assertEqual(second["left"]["gap_count"], 0)

    def test_post_stop_recovery_cannot_be_recorded_inside_episode(self):
        clock = _BlockingFinishClock()
        tracker = Dex3SubscriberDropTracker(clock_ns=clock)
        tracker.observe("left", received_ns=0)
        tracker.begin_episode_window()

        result = []
        finish_thread = threading.Thread(
            target=lambda: result.append(tracker.finish_episode_window())
        )
        recovery_thread = threading.Thread(
            target=lambda: tracker.observe("left", received_ns=2_000_000_000)
        )
        finish_thread.start()
        self.assertTrue(clock.finish_clock_entered.wait(2.0))
        recovery_thread.start()
        clock.release_finish_clock.set()
        finish_thread.join(2.0)
        recovery_thread.join(2.0)

        self.assertFalse(finish_thread.is_alive())
        self.assertFalse(recovery_thread.is_alive())
        self.assertEqual(result[0]["left"]["recovered_gap_count"], 0)
        self.assertTrue(result[0]["left"]["open_gap_at_end"])
        self.assertAlmostEqual(result[0]["left"]["gaps"][0]["duration_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
