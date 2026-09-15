import threading
import unittest

from teleop.utils.episode_voice_feedback import (
    RECORDING_SAVED,
    STARTING_RECORDING,
    STOPPING_RECORDING,
    AsyncEpisodeVoiceNotifier,
    EpisodeRecordingController,
)


class _Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Recorder:
    def __init__(self, *, create_result=True):
        self.create_result = create_result
        self.ready = True
        self.create_calls = 0
        self.save_calls = 0

    def create_episode(self):
        self.create_calls += 1
        if self.create_result:
            self.ready = False
        return self.create_result

    def save_episode(self):
        self.save_calls += 1

    def is_ready(self):
        return self.ready

    def complete_save(self):
        self.ready = True


class _Notifier:
    def __init__(self, *, raise_on_notify=False):
        self.messages = []
        self.raise_on_notify = raise_on_notify

    def notify(self, message):
        if self.raise_on_notify:
            raise RuntimeError("speaker unavailable")
        self.messages.append(message)
        return True


class TestEpisodeRecordingController(unittest.TestCase):
    def make_controller(self, *, create_result=True, notifier=None):
        clock = _Clock()
        recorder = _Recorder(create_result=create_result)
        notifier = notifier or _Notifier()
        controller = EpisodeRecordingController(
            recorder,
            notifier=notifier,
            debounce_s=0.5,
            clock=clock,
        )
        return controller, recorder, notifier, clock

    def start_recording(self, controller):
        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "started")
        self.assertTrue(controller.recording)

    def release_and_rearm(self, controller, clock):
        self.assertTrue(controller.request_key_release())
        clock.advance(0.5)

    def test_start_announces_only_after_successful_create(self):
        controller, recorder, notifier, _clock = self.make_controller()

        self.start_recording(controller)

        self.assertEqual(recorder.create_calls, 1)
        self.assertEqual(notifier.messages, [STARTING_RECORDING])

    def test_failed_create_does_not_announce_start(self):
        controller, recorder, notifier, _clock = self.make_controller(
            create_result=False
        )

        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "start_failed")

        self.assertFalse(controller.recording)
        self.assertEqual(recorder.create_calls, 1)
        self.assertEqual(notifier.messages, [])

    def test_duplicate_or_held_s_cannot_stop_just_started_episode(self):
        controller, recorder, notifier, clock = self.make_controller()

        self.assertTrue(controller.request_key_press())
        self.assertFalse(controller.request_key_press())
        self.assertEqual(controller.update(), "started")
        clock.advance(5.0)
        self.assertFalse(controller.request_key_press())
        self.assertIsNone(controller.update())

        self.assertTrue(controller.recording)
        self.assertEqual(recorder.save_calls, 0)
        self.assertEqual(notifier.messages, [STARTING_RECORDING])

    def test_release_must_be_followed_by_debounce_before_stop(self):
        controller, recorder, _notifier, clock = self.make_controller()
        self.start_recording(controller)

        self.assertTrue(controller.request_key_release())
        clock.advance(0.499)
        self.assertFalse(controller.request_key_press())
        self.assertTrue(controller.request_key_release())
        clock.advance(0.5)
        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "stopping")

        self.assertFalse(controller.recording)
        self.assertEqual(recorder.save_calls, 1)

    def test_stop_is_single_edge_and_save_pending_rejects_all_s(self):
        controller, recorder, notifier, clock = self.make_controller()
        self.start_recording(controller)
        self.release_and_rearm(controller, clock)

        self.assertTrue(controller.request_key_press())
        self.assertFalse(controller.request_key_press())
        self.assertEqual(controller.update(), "stopping")
        self.assertTrue(controller.save_pending)
        self.assertEqual(recorder.save_calls, 1)
        self.assertEqual(
            notifier.messages,
            [STARTING_RECORDING, STOPPING_RECORDING],
        )

        self.assertFalse(controller.request_key_press())
        self.assertTrue(controller.request_key_release())
        clock.advance(10.0)
        self.assertFalse(controller.request_key_press())
        self.assertTrue(controller.request_key_release())
        self.assertIsNone(controller.update())
        self.assertEqual(recorder.create_calls, 1)
        self.assertEqual(recorder.save_calls, 1)

    def test_saved_announces_only_after_writer_ready(self):
        controller, recorder, notifier, clock = self.make_controller()
        self.start_recording(controller)
        self.release_and_rearm(controller, clock)
        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "stopping")

        self.assertFalse(controller.poll_save_completion())
        self.assertNotIn(RECORDING_SAVED, notifier.messages)

        recorder.complete_save()
        self.assertTrue(controller.poll_save_completion())
        self.assertFalse(controller.poll_save_completion())
        self.assertEqual(notifier.messages.count(RECORDING_SAVED), 1)

    def test_held_s_across_save_completion_requires_release_and_debounce(self):
        controller, recorder, notifier, clock = self.make_controller()
        self.start_recording(controller)
        self.release_and_rearm(controller, clock)
        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "stopping")

        recorder.complete_save()
        self.assertEqual(controller.update(), "saved")
        self.assertFalse(controller.ready_to_start)
        clock.advance(5.0)
        self.assertFalse(controller.request_key_press())
        self.assertFalse(controller.ready_to_start)

        self.assertTrue(controller.request_key_release())
        clock.advance(0.499)
        self.assertFalse(controller.ready_to_start)
        clock.advance(0.001)
        self.assertTrue(controller.ready_to_start)
        self.assertEqual(notifier.messages[-1], RECORDING_SAVED)

    def test_ipc_tap_is_debounced_and_ignored_while_save_pending(self):
        controller, recorder, _notifier, clock = self.make_controller()

        self.assertTrue(controller.request_tap())
        self.assertEqual(controller.update(), "started")
        self.assertFalse(controller.request_tap())
        self.assertIsNone(controller.update())

        clock.advance(0.5)
        self.assertTrue(controller.request_tap())
        self.assertEqual(controller.update(), "stopping")
        clock.advance(10.0)
        self.assertFalse(controller.request_tap())

        recorder.complete_save()
        self.assertEqual(controller.update(), "saved")
        self.assertEqual(recorder.create_calls, 1)
        self.assertEqual(recorder.save_calls, 1)

    def test_notifier_exception_is_fail_open(self):
        notifier = _Notifier(raise_on_notify=True)
        controller, recorder, _notifier, _clock = self.make_controller(
            notifier=notifier
        )

        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "started")
        self.assertTrue(controller.recording)
        self.assertEqual(recorder.create_calls, 1)

    def test_active_shutdown_stops_and_saves_exactly_once(self):
        controller, recorder, notifier, _clock = self.make_controller()
        self.start_recording(controller)

        self.assertTrue(controller.stop_for_shutdown())
        self.assertFalse(controller.recording)
        self.assertTrue(controller.save_pending)
        self.assertEqual(recorder.save_calls, 1)
        self.assertEqual(
            notifier.messages,
            [STARTING_RECORDING, STOPPING_RECORDING],
        )

        self.assertFalse(controller.stop_for_shutdown())
        self.assertEqual(recorder.save_calls, 1)
        self.assertEqual(notifier.messages.count(STOPPING_RECORDING), 1)

        recorder.complete_save()
        self.assertTrue(controller.poll_save_completion())
        self.assertEqual(notifier.messages[-1], RECORDING_SAVED)

    def test_pending_save_shutdown_does_not_request_or_announce_stop_twice(self):
        controller, recorder, notifier, clock = self.make_controller()
        self.start_recording(controller)
        self.release_and_rearm(controller, clock)
        self.assertTrue(controller.request_key_press())
        self.assertEqual(controller.update(), "stopping")

        self.assertFalse(controller.stop_for_shutdown())
        self.assertEqual(recorder.save_calls, 1)
        self.assertEqual(notifier.messages.count(STOPPING_RECORDING), 1)

        recorder.complete_save()
        self.assertTrue(controller.poll_save_completion())
        self.assertEqual(notifier.messages.count(RECORDING_SAVED), 1)


class TestAsyncEpisodeVoiceNotifier(unittest.TestCase):
    def test_speaker_runs_only_on_background_worker(self):
        spoken = []
        spoke = threading.Event()

        def speaker(message):
            spoken.append((message, threading.current_thread().name))
            spoke.set()

        notifier = AsyncEpisodeVoiceNotifier(speaker=speaker)
        try:
            self.assertTrue(notifier.notify(STARTING_RECORDING))
            self.assertTrue(spoke.wait(timeout=1.0))
        finally:
            notifier.close()

        self.assertEqual(spoken, [(STARTING_RECORDING, "episode-voice-feedback")])

    def test_queue_is_bounded_and_full_queue_never_blocks_caller(self):
        speaking = threading.Event()
        release = threading.Event()

        def speaker(_message):
            speaking.set()
            release.wait(timeout=1.0)

        notifier = AsyncEpisodeVoiceNotifier(
            speaker=speaker,
            queue_capacity=1,
        )
        try:
            self.assertTrue(notifier.notify("one"))
            self.assertTrue(speaking.wait(timeout=1.0))
            self.assertTrue(notifier.notify("two"))
            self.assertFalse(notifier.notify("three"))
            self.assertEqual(notifier.dropped_count, 1)
        finally:
            release.set()
            notifier.close()

    def test_close_normally_flushes_recording_saved(self):
        spoken = []
        notifier = AsyncEpisodeVoiceNotifier(speaker=spoken.append)

        self.assertTrue(notifier.notify(RECORDING_SAVED))
        notifier.close()

        self.assertEqual(spoken, [RECORDING_SAVED])

    def test_speaker_failure_disables_feedback_without_raising_to_caller(self):
        attempted = threading.Event()

        def speaker(_message):
            attempted.set()
            raise RuntimeError("no audio device")

        notifier = AsyncEpisodeVoiceNotifier(speaker=speaker)
        try:
            self.assertTrue(notifier.notify("one"))
            self.assertTrue(attempted.wait(timeout=1.0))
            notifier._worker.join(timeout=1.0)
            self.assertFalse(notifier.available)
            self.assertFalse(notifier.notify("two"))
        finally:
            notifier.close()


if __name__ == "__main__":
    unittest.main()
