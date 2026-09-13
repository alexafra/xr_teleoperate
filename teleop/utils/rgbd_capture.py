"""Experimental head-camera atomic RGBD selection for recording.

When explicitly enabled in code, the atomic TeleImager stream is preferred
because its colour and aligned-depth images are encoded from one camera
frameset.  The checked-in default is disabled, and older or temporarily
unhealthy servers remain usable through legacy independent streams.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


RGBD_PAIRING_SCHEMA_VERSION = 1
RGBD_PROTOCOL = "teleimager-rgbd-v1"
LEGACY_RGBD_PROTOCOL = "legacy-independent-zmq"
INSPIRE_END_EFFECTORS = frozenset({"inspire_dfx", "inspire_ftp"})
EXPERIMENTAL_ATOMIC_RGBD_RECORDING = False


def should_prefer_atomic_rgbd_recording(
    *,
    recording_enabled: bool,
    end_effector: Optional[str],
    head_config: Mapping[str, Any],
    experimental_opt_in: bool = EXPERIMENTAL_ATOMIC_RGBD_RECORDING,
) -> bool:
    """Keep atomic recording dormant unless explicitly enabled in code."""

    return bool(
        experimental_opt_in
        and recording_enabled
        and end_effector in INSPIRE_END_EFFECTORS
        and head_config.get("enable_depth", False)
    )


@dataclass(frozen=True)
class HeadRgbdSample:
    """One decoded head-camera sample and its pairing provenance."""

    color_bgr: Any
    aligned_depth: Any
    pairing: dict[str, Any]


class PreferAtomicHeadRgbdCapture:
    """Prefer atomic RGBD and transparently fall back to legacy streams.

    Atomic receive/decode problems are contained here so they cannot terminate
    arm or hand control.  A repeated atomic frame is normal when the control
    loop runs faster than the camera, so a short grace period is allowed before
    declaring the stream stalled.  Before the first atomic success, legacy mode
    periodically probes it to tolerate ZeroMQ slow-joiner startup.  A failure
    after atomic recording has begun makes legacy fallback sticky for the
    process, avoiding transport flapping and repeated transition warnings.
    """

    def __init__(
        self,
        image_client: Any,
        head_config: Mapping[str, Any],
        *,
        decode_atomic_frame: Optional[Callable[[Any], tuple[Any, Any]]],
        logger: Any = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        stall_timeout_s: float = 0.15,
        recovery_probe_interval_s: float = 1.0,
    ) -> None:
        if stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be positive")
        if recovery_probe_interval_s < 0:
            raise ValueError("recovery_probe_interval_s must be non-negative")

        self._image_client = image_client
        self._head_config = dict(head_config)
        self._decode_atomic_frame = decode_atomic_frame
        self._logger = logger or logging.getLogger(__name__)
        self._monotonic_ns = monotonic_ns
        self._stall_timeout_ns = int(stall_timeout_s * 1_000_000_000)
        self._recovery_probe_interval_ns = int(
            recovery_probe_interval_s * 1_000_000_000
        )

        self._atomic_getter = getattr(image_client, "get_head_rgbd_frame", None)
        self._atomic_unavailable_reason = self._get_atomic_unavailable_reason()
        self._mode: Optional[str] = None
        self._legacy_reason = self._atomic_unavailable_reason
        self._next_atomic_probe_ns: Optional[int] = None
        self._last_atomic_sequence: Optional[int] = None
        self._last_atomic_received_ns: Optional[int] = None
        self._last_atomic_sample: Optional[HeadRgbdSample] = None
        self._consecutive_sample_holds = 0
        self._atomic_has_succeeded = False

    @property
    def mode(self) -> Optional[str]:
        """Current transport mode, once the first sample has been requested."""

        return self._mode

    @property
    def atomic_capable(self) -> bool:
        """Whether configuration and client API advertise the v1 stream."""

        return self._atomic_unavailable_reason is None

    def episode_metadata(self) -> dict[str, Any]:
        """Describe the selection policy and per-frame provenance contract."""

        return {
            "schema_version": RGBD_PAIRING_SCHEMA_VERSION,
            "experimental": True,
            "selection": "prefer_atomic_with_legacy_fallback",
            "preferred_protocol": RGBD_PROTOCOL,
            "per_frame_field": "rgbd_pairing",
            "atomic_pairing": "same_server_capture",
            "legacy_pairing": "unverified_independent_receives",
            "raw_depth_pairing": "independent_legacy_stream",
            "atomic_capable_at_startup": self.atomic_capable,
        }

    def read(self) -> HeadRgbdSample:
        """Return a new/held atomic sample or the current legacy sample.

        During the brief grace period after a healthy atomic stream repeats or
        has not produced its next frame, the last decoded pair is returned with
        explicit hold provenance.  This keeps the episode row cadence stable.
        """

        now_ns = self._monotonic_ns()
        failure_reason = self._atomic_unavailable_reason
        failure_detail = None
        atomic_probe_attempted = False

        if failure_reason is None and self._atomic_probe_is_due(now_ns):
            atomic_probe_attempted = True
            atomic_sample, failure_reason, failure_detail = self._read_atomic()
            if atomic_sample is not None:
                self._set_atomic_mode()
                return atomic_sample

            if (
                failure_reason in {"atomic_no_frame", "atomic_duplicate"}
                and self._mode == "atomic"
                and self._last_atomic_received_ns is not None
                and now_ns - self._last_atomic_received_ns
                <= self._stall_timeout_ns
                and self._last_atomic_sample is not None
            ):
                return self._hold_last_atomic_sample()

            if (
                failure_reason in {"atomic_no_frame", "atomic_duplicate"}
                and self._mode == "atomic"
            ):
                failure_reason = "atomic_stalled"

        if failure_reason is None:
            failure_reason = self._legacy_reason or "atomic_unavailable"

        return self._read_legacy(
            now_ns,
            reason=failure_reason,
            detail=failure_detail,
            atomic_probe_attempted=atomic_probe_attempted,
        )

    def _get_atomic_unavailable_reason(self) -> Optional[str]:
        if not self._head_config.get("enable_depth", False):
            return "depth_disabled"

        rgbd_port = self._head_config.get("rgbd_zmq_port")
        if (
            isinstance(rgbd_port, bool)
            or not isinstance(rgbd_port, int)
            or not 1 <= rgbd_port <= 65535
        ):
            return "atomic_port_not_configured"

        if self._head_config.get("rgbd_protocol") != RGBD_PROTOCOL:
            return "atomic_protocol_unsupported"
        if not callable(self._atomic_getter):
            return "atomic_client_api_unavailable"
        if not callable(self._decode_atomic_frame):
            return "atomic_decoder_unavailable"
        return None

    def _atomic_probe_is_due(self, now_ns: int) -> bool:
        if self._mode != "legacy":
            return True
        if self._atomic_has_succeeded:
            return False
        return (
            self._next_atomic_probe_ns is None
            or now_ns >= self._next_atomic_probe_ns
        )

    def _read_atomic(
        self,
    ) -> tuple[Optional[HeadRgbdSample], Optional[str], Optional[str]]:
        try:
            frame = self._atomic_getter()
        except Exception as exc:
            return None, "atomic_receive_error", self._exception_detail(exc)

        if frame is None:
            return None, "atomic_no_frame", None

        try:
            sequence = self._validate_nonnegative_int(
                "sequence",
                frame.sequence,
            )
            server_capture_ns = self._validate_nonnegative_int(
                "server_capture_monotonic_ns",
                frame.server_capture_monotonic_ns,
            )
            received_ns = self._validate_nonnegative_int(
                "received_monotonic_ns",
                frame.received_monotonic_ns,
            )
            # The subscriber can receive a new packet while the getter runs,
            # so check freshness against a clock sample taken afterwards.
            age_ns = self._monotonic_ns() - received_ns
            if age_ns < 0:
                raise ValueError("received_monotonic_ns is in the future")
            if age_ns > self._stall_timeout_ns:
                return None, "atomic_stale_frame", None
        except Exception as exc:
            return None, "atomic_invalid_metadata", self._exception_detail(exc)

        if sequence == self._last_atomic_sequence:
            return None, "atomic_duplicate", None

        try:
            color_bgr, aligned_depth = self._decode_atomic_frame(frame)
            if color_bgr is None or aligned_depth is None:
                raise ValueError("atomic decoder returned an empty image")
        except Exception as exc:
            return None, "atomic_decode_error", self._exception_detail(exc)

        sequence_regressed = (
            self._last_atomic_sequence is not None
            and sequence < self._last_atomic_sequence
        )
        sequence_gap = 0
        if (
            self._last_atomic_sequence is not None
            and sequence > self._last_atomic_sequence + 1
        ):
            sequence_gap = sequence - self._last_atomic_sequence - 1

        self._last_atomic_sequence = sequence
        self._last_atomic_received_ns = received_ns

        pairing = {
            "schema_version": RGBD_PAIRING_SCHEMA_VERSION,
            "mode": "atomic",
            "paired": True,
            "protocol": RGBD_PROTOCOL,
            "capture_sequence": sequence,
            "server_capture_monotonic_ns": server_capture_ns,
            "client_received_monotonic_ns": received_ns,
            "sample_held": False,
            "sample_hold_count": 0,
            "raw_depth_pairing": "independent_legacy_stream",
        }
        if sequence_gap:
            pairing["capture_sequence_gap"] = sequence_gap
        if sequence_regressed:
            pairing["capture_sequence_regressed"] = True

        sample = HeadRgbdSample(
            color_bgr=color_bgr,
            aligned_depth=aligned_depth,
            pairing=pairing,
        )
        self._last_atomic_sample = sample
        self._consecutive_sample_holds = 0
        return (
            sample,
            None,
            None,
        )

    def _hold_last_atomic_sample(self) -> HeadRgbdSample:
        self._consecutive_sample_holds += 1
        pairing = dict(self._last_atomic_sample.pairing)
        pairing["sample_held"] = True
        pairing["sample_hold_count"] = self._consecutive_sample_holds
        return HeadRgbdSample(
            color_bgr=self._last_atomic_sample.color_bgr,
            aligned_depth=self._last_atomic_sample.aligned_depth,
            pairing=pairing,
        )

    def _read_legacy(
        self,
        now_ns: int,
        *,
        reason: str,
        detail: Optional[str],
        atomic_probe_attempted: bool,
    ) -> HeadRgbdSample:
        self._set_legacy_mode(
            now_ns,
            reason=reason,
            detail=detail,
            atomic_probe_attempted=atomic_probe_attempted,
        )

        head_frame = self._image_client.get_head_frame()
        color_bgr = (
            None if head_frame is None else getattr(head_frame, "bgr", None)
        )
        aligned_depth = self._image_client.get_head_depth_frame()
        return HeadRgbdSample(
            color_bgr=color_bgr,
            aligned_depth=aligned_depth,
            pairing={
                "schema_version": RGBD_PAIRING_SCHEMA_VERSION,
                "mode": "legacy",
                "paired": False,
                "protocol": LEGACY_RGBD_PROTOCOL,
                "fallback_reason": reason,
                "raw_depth_pairing": "independent_legacy_stream",
            },
        )

    def _set_atomic_mode(self) -> None:
        if self._mode != "atomic":
            self._safe_log(
                "info",
                "Atomic RGBD active; recording capture-synchronised colour "
                "and aligned depth.",
            )
        self._mode = "atomic"
        self._atomic_has_succeeded = True
        self._legacy_reason = None
        self._next_atomic_probe_ns = None

    def _set_legacy_mode(
        self,
        now_ns: int,
        *,
        reason: str,
        detail: Optional[str],
        atomic_probe_attempted: bool,
    ) -> None:
        entering_legacy = self._mode != "legacy"
        if entering_legacy:
            detail_suffix = f" ({detail})" if detail else ""
            self._safe_log(
                "warning",
                "Atomic RGBD unavailable; using legacy independent colour/"
                "depth streams, whose pairing is unverified. "
                f"Reason: {reason}{detail_suffix}",
            )
        self._mode = "legacy"
        self._legacy_reason = reason
        if self._atomic_has_succeeded:
            self._next_atomic_probe_ns = None
        elif entering_legacy or atomic_probe_attempted:
            self._next_atomic_probe_ns = (
                now_ns + self._recovery_probe_interval_ns
            )

    def _safe_log(self, level: str, message: str) -> None:
        try:
            getattr(self._logger, level)(message)
        except Exception:
            # Logging an atomic transport transition must not affect control.
            pass

    @staticmethod
    def _validate_nonnegative_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _exception_detail(exc: Exception) -> str:
        message = str(exc).strip()
        if not message:
            return type(exc).__name__
        return f"{type(exc).__name__}: {message}"
