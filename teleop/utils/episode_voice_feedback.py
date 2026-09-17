from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import subprocess
import threading
import time

import logging_mp


logger_mp = logging_mp.getLogger(__name__)

STARTING_RECORDING = "Starting recording"
STOPPING_RECORDING = "Stopping recording"
RECORDING_SAVED = "Recording saved"

_PROMPT_FILENAMES = {
    STARTING_RECORDING: "starting_recording.wav",
    STOPPING_RECORDING: "stopping_recording.wav",
    RECORDING_SAVED: "recording_saved.wav",
}
_DEFAULT_PROMPT_DIR = Path(__file__).with_name("episode_voice_prompts")

_STOP = object()


class AsyncEpisodeVoiceNotifier:
    """Serialize speech outside the teleoperation/control thread."""

    def __init__(
        self,
        *,
        queue_capacity=8,
        speaker=None,
        executable=None,
        audio_player=None,
        prompt_dir=None,
        speech_timeout_s=15.0,
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if speech_timeout_s <= 0:
            raise ValueError("speech_timeout_s must be positive")

        self._queue = Queue(maxsize=queue_capacity)
        self._speaker = speaker
        self._speech_timeout_s = float(speech_timeout_s)
        self._force_stop = threading.Event()
        self._closed = False
        self._failed = False
        self._dropped_count = 0
        self._warning_logged = False
        self._state_lock = threading.Lock()
        self._prompt_dir = Path(prompt_dir) if prompt_dir is not None else _DEFAULT_PROMPT_DIR

        if speaker is None:
            self._audio_player = audio_player or shutil.which("pw-play")
            self._executable = executable or shutil.which("spd-say")
            self._natural_voice_available = self._audio_player is not None and all(
                (self._prompt_dir / filename).is_file()
                for filename in _PROMPT_FILENAMES.values()
            )
            if not self._natural_voice_available and self._executable is None:
                self._failed = True
                self._warn_once(
                    "Episode voice feedback is disabled because the bundled "
                    "voice prompts cannot be played and spd-say is not installed."
                )
        else:
            self._audio_player = audio_player
            self._executable = executable
            self._natural_voice_available = False

        self._worker = None
        if not self._failed:
            self._worker = threading.Thread(
                target=self._run,
                name="episode-voice-feedback",
                daemon=True,
            )
            self._worker.start()

    @property
    def available(self):
        with self._state_lock:
            return not self._failed and not self._closed

    @property
    def dropped_count(self):
        with self._state_lock:
            return self._dropped_count

    def _warn_once(self, message):
        with self._state_lock:
            if self._warning_logged:
                return
            self._warning_logged = True
        logger_mp.warning(message)

    def notify(self, message):
        """Queue a phrase without waiting for speech or queue capacity."""
        with self._state_lock:
            if self._closed or self._failed:
                return False
            try:
                self._queue.put_nowait(str(message))
            except Full:
                # This can be reached from the 30 Hz loop. Accounting is kept
                # in-memory and reported during close; never log or block here.
                self._dropped_count += 1
                return False
        return True

    def _run_speech_process(self, command, description):
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + self._speech_timeout_s
        while process.poll() is None:
            if self._force_stop.wait(0.05):
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
                return
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
                raise TimeoutError(f"{description} did not finish before its timeout")

        if process.returncode != 0:
            raise RuntimeError(f"{description} exited with status {process.returncode}")

    def _speak_message(self, message):
        prompt_filename = _PROMPT_FILENAMES.get(message)
        if self._natural_voice_available and prompt_filename is not None:
            self._run_speech_process(
                [self._audio_player, str(self._prompt_dir / prompt_filename)],
                "natural voice playback",
            )
            return

        self._run_speech_process(
            [self._executable, "--wait", message],
            "spd-say",
        )

    def _run(self):
        while True:
            try:
                message = self._queue.get(timeout=0.1)
            except Empty:
                if self._force_stop.is_set():
                    return
                continue

            try:
                if message is _STOP:
                    return
                if self._failed:
                    continue
                if self._speaker is None:
                    self._speak_message(message)
                else:
                    self._speaker(message)
            except Exception as error:
                with self._state_lock:
                    self._failed = True
                self._warn_once(
                    f"Episode voice feedback failed and has been disabled: {error}"
                )
            finally:
                self._queue.task_done()

    def close(self, timeout_s=10.0):
        """Stop accepting messages and bound shutdown time."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker

        if worker is None:
            return

        try:
            self._queue.put(_STOP, timeout=0.1)
        except Full:
            self._force_stop.set()

        worker.join(timeout=max(0.0, float(timeout_s)))
        if worker.is_alive():
            self._force_stop.set()
            worker.join(timeout=1.0)

        with self._state_lock:
            dropped_count = self._dropped_count
        if dropped_count:
            logger_mp.warning(
                "Episode voice feedback dropped "
                f"{dropped_count} announcement(s) because its queue was full."
            )


class EpisodeRecordingController:
    """Edge-trigger recording transitions with async-save interlocking."""

    def __init__(
        self,
        recorder,
        *,
        notifier=None,
        debounce_s=0.5,
        clock=time.monotonic,
    ):
        if debounce_s < 0:
            raise ValueError("debounce_s must not be negative")
        self._recorder = recorder
        self._notifier = notifier
        self._debounce_s = float(debounce_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._recording = False
        self._save_pending = False
        self._transition_in_progress = False
        self._toggle_requested = False
        self._key_down = False
        self._accept_after = 0.0

    @property
    def recording(self):
        with self._lock:
            return self._recording

    @property
    def save_pending(self):
        with self._lock:
            return self._save_pending

    @property
    def ready_to_start(self):
        now = self._clock()
        with self._lock:
            return (
                not self._recording
                and not self._save_pending
                and not self._transition_in_progress
                and not self._toggle_requested
                and not self._key_down
                and now >= self._accept_after
            )

    def request_key_press(self):
        """Accept one S-down edge; repeated/held S events are ignored."""
        now = self._clock()
        with self._lock:
            if self._key_down:
                return False
            self._key_down = True
            if (
                self._save_pending
                or self._transition_in_progress
                or self._toggle_requested
                or now < self._accept_after
            ):
                return False
            self._toggle_requested = True
            self._accept_after = now + self._debounce_s
            return True

    def request_key_release(self):
        """Re-arm only after a real release followed by the debounce period."""
        now = self._clock()
        with self._lock:
            was_down = self._key_down
            self._key_down = False
            if was_down:
                self._accept_after = max(
                    self._accept_after,
                    now + self._debounce_s,
                )
            return was_down

    def request_tap(self):
        """Treat an IPC record command as a complete press/release edge."""
        accepted = self.request_key_press()
        self.request_key_release()
        return accepted

    def _notify(self, message):
        if self._notifier is not None:
            try:
                return self._notifier.notify(message)
            except Exception:
                # Voice feedback is advisory and cannot interrupt recording
                # or the robot control loop.
                return False
        return False

    def poll_save_completion(self):
        """Announce a save only after EpisodeWriter reports completion."""
        with self._lock:
            if not self._save_pending:
                return False

        if not self._recorder.is_ready():
            return False

        with self._lock:
            if not self._save_pending:
                return False
            self._save_pending = False
        self._notify(RECORDING_SAVED)
        return True

    def stop_for_shutdown(self):
        """Stop one active episode during exit without duplicating a pending save."""
        with self._lock:
            # A queued S edge must not be applied after shutdown has begun.
            self._toggle_requested = False
            if self._save_pending or not self._recording:
                return False
            self._recording = False
            self._save_pending = True
            self._transition_in_progress = True

        self._notify(STOPPING_RECORDING)
        try:
            self._recorder.save_episode()
        finally:
            with self._lock:
                self._transition_in_progress = False
        return True

    def update(self):
        """Apply at most one accepted transition from the control thread."""
        if self.poll_save_completion():
            completed = True
        else:
            completed = False

        with self._lock:
            if not self._toggle_requested:
                return "saved" if completed else None
            self._toggle_requested = False
            self._transition_in_progress = True
            stopping = self._recording
            if stopping:
                self._recording = False
                self._save_pending = True

        if stopping:
            self._notify(STOPPING_RECORDING)
            try:
                self._recorder.save_episode()
            finally:
                with self._lock:
                    self._transition_in_progress = False
            return "stopping"

        try:
            created = bool(self._recorder.create_episode())
        finally:
            with self._lock:
                self._transition_in_progress = False

        if not created:
            return "start_failed"

        with self._lock:
            self._recording = True
        self._notify(STARTING_RECORDING)
        return "started"
