import shutil
import subprocess

import logging_mp


logger_mp = logging_mp.getLogger(__name__)


class VoiceFeedback:
    """Best-effort spoken status messages via Speech Dispatcher."""

    def __init__(self, enabled=False):
        self.command = shutil.which("spd-say") if enabled else None
        if enabled and self.command is None:
            logger_mp.warning(
                "Voice feedback requested, but 'spd-say' is not installed."
            )

    def say(self, message):
        if self.command is None:
            return

        try:
            # Speech Dispatcher queues the message; do not block the control loop
            # while the audio is playing.
            subprocess.Popen(
                [self.command, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger_mp.warning(f"Failed to play voice feedback: {exc}")
