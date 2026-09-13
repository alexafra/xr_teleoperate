"""Subscribe to Inspire FTP/E2 tactile feedback for episode recording.

The bridge publishes tactile feedback at roughly 20 Hz while episodes are
normally recorded at 30 Hz, so adjacent frames can legitimately contain the
same latest sample. This module creates subscribers only; it cannot command
the hands.
"""

import threading
import time

import logging_mp


logger_mp = logging_mp.getLogger(__name__)

kTopicInspireFTPLeftTouch = "rt/inspire_hand/touch/l"
kTopicInspireFTPRightTouch = "rt/inspire_hand/touch/r"

# Ordered exactly as inspire_hand_touch.idl.
TACTILE_PAD_LENGTHS = {
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
TACTILE_PADS = tuple(TACTILE_PAD_LENGTHS)
TACTILE_CELLS_PER_HAND = sum(TACTILE_PAD_LENGTHS.values())
TACTILE_STALE_AFTER_S = 0.25


class InspireTactileReader:
    """Keep the latest validated tactile frame from each FTP hand."""

    def __init__(
        self,
        *,
        subscriber_factory=None,
        touch_message_type=None,
        clock_ns=time.monotonic_ns,
        stale_after_s=TACTILE_STALE_AFTER_S,
    ):
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        if subscriber_factory is None:
            from unitree_sdk2py.core.channel import ChannelSubscriber

            subscriber_factory = ChannelSubscriber
        if touch_message_type is None:
            try:
                from inspire_sdkpy import inspire_dds
            except ImportError as error:
                raise RuntimeError(
                    "Inspire tactile capture requires the vendor inspire_sdkpy package"
                ) from error
            touch_message_type = inspire_dds.inspire_hand_touch

        self._clock_ns = clock_ns
        self._stale_after_ns = int(float(stale_after_s) * 1e9)
        self._lock = threading.Lock()
        self._latest = {"left": None, "right": None}
        self._received_at_ns = {"left": None, "right": None}
        self._counts = {"left": 0, "right": 0}
        self._stale = False
        self._closed = False
        self._subscribers = []

        try:
            for side, topic in (
                ("left", kTopicInspireFTPLeftTouch),
                ("right", kTopicInspireFTPRightTouch),
            ):
                subscriber = subscriber_factory(topic, touch_message_type)
                self._subscribers.append(subscriber)
                subscriber.Init(self._make_callback(side), 10)
        except Exception:
            self.close()
            raise

        logger_mp.info(
            "[InspireTactileReader] subscribed to %s and %s",
            kTopicInspireFTPLeftTouch,
            kTopicInspireFTPRightTouch,
        )

    @staticmethod
    def _decode_message(message):
        decoded = {}
        for pad, expected_length in TACTILE_PAD_LENGTHS.items():
            try:
                values = list(getattr(message, pad))
            except (AttributeError, TypeError) as error:
                raise ValueError(f"missing or invalid tactile pad {pad!r}") from error
            if len(values) != expected_length:
                raise ValueError(
                    f"tactile pad {pad!r} has {len(values)} cells; expected {expected_length}"
                )
            decoded[pad] = tuple(int(value) for value in values)
        return decoded

    def _make_callback(self, side):
        def callback(message):
            try:
                decoded = self._decode_message(message)
            except (TypeError, ValueError) as error:
                logger_mp.warning(
                    "[InspireTactileReader] ignored invalid %s tactile frame: %s",
                    side,
                    error,
                )
                return
            received_at_ns = self._clock_ns()
            with self._lock:
                if self._closed:
                    return
                self._latest[side] = decoded
                self._received_at_ns[side] = received_at_ns
                self._counts[side] += 1

        return callback

    def received(self):
        with self._lock:
            return dict(self._counts)

    def _both_sides_fresh(self):
        now_ns = self._clock_ns()
        with self._lock:
            return all(
                self._latest[side] is not None
                and self._received_at_ns[side] is not None
                and now_ns - self._received_at_ns[side] <= self._stale_after_ns
                for side in ("left", "right")
            )

    def wait_for_data(self, timeout=3.0):
        """Return once both hands have supplied one complete, valid frame."""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self._both_sides_fresh():
                return True
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        counts = self.received()
        if self._both_sides_fresh():
            return True
        logger_mp.warning(
            "[InspireTactileReader] no complete tactile pair after %.2fs "
            "(left=%d, right=%d). Check Headless_driver_double.py and both touch topics.",
            float(timeout),
            counts["left"],
            counts["right"],
        )
        return False

    def get_tactiles(self):
        """Return a detached snapshot, or ``None`` while either side is stale."""

        now_ns = self._clock_ns()
        with self._lock:
            unavailable = [
                side
                for side in ("left", "right")
                if self._latest[side] is None
                or self._received_at_ns[side] is None
                or now_ns - self._received_at_ns[side] > self._stale_after_ns
            ]
            if unavailable:
                if not self._stale:
                    logger_mp.warning(
                        "[InspireTactileReader] tactile data unavailable/stale for %s; "
                        "recording tactiles=null until both sides recover",
                        ", ".join(unavailable),
                    )
                self._stale = True
                return None
            if self._stale:
                logger_mp.info("[InspireTactileReader] both tactile streams recovered")
            self._stale = False
            return {
                "left_ee": {
                    pad: list(self._latest["left"][pad])
                    for pad in TACTILE_PADS
                },
                "right_ee": {
                    pad: list(self._latest["right"][pad])
                    for pad in TACTILE_PADS
                },
            }

    def close(self):
        """Close both subscribers. Safe to call more than once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            try:
                subscriber.Close()
            except Exception as error:
                logger_mp.warning(
                    "[InspireTactileReader] failed to close a subscriber: %s",
                    error,
                )
