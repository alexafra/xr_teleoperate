"""Read-only quality classification for raw teleoperation episodes.

The scanner never changes an episode.  It only creates a manifest when an
explicit output path is supplied, and refuses to overwrite an existing file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_FRAME_GAP_S = 0.075
DEFAULT_MIN_MEASURED_FPS = 29.0

_DFX_COUNTER_KEYS = {
    "baseline_count",
    "held_sample_count",
    "drop_event_count",
    "lost_increment_count",
    "reset_count",
    "counter_regression_count",
    "counter_divergence_count",
    "malformed_message_count",
    "total_drop_event_count",
    "total_held_sample_count",
    "total_lost_increment_count",
    "total_reset_count",
}

_INSPIRE_JOINT_NAMES = (
    "Pinky",
    "Ring",
    "Middle",
    "Index",
    "ThumbBend",
    "ThumbRotation",
)

_EXPECTED_INSPIRE_END_EFFECTOR_BASE = {
    "schema_version": 1,
    "type": "inspire",
    "hand_dof": 6,
    "value_unit": "normalized_open_fraction",
    "value_range": [0.0, 1.0],
    "zero_semantics": "fully_closed",
    "one_semantics": "fully_open",
    "left_joint_names": [f"kLeftHand{name}" for name in _INSPIRE_JOINT_NAMES],
    "right_joint_names": [f"kRightHand{name}" for name in _INSPIRE_JOINT_NAMES],
    "canonical_order": "left_then_right",
}

_DDS_SOURCE_CONTRACTS = {
    "dex3": (
        "dex3_state_subscribers",
        {
            "left": "rt/dex3/left/state",
            "right": "rt/dex3/right/state",
        },
    ),
    "inspire_ftp": (
        "inspire_ftp_state_subscribers",
        {
            "left": "rt/inspire_hand/state/l",
            "right": "rt/inspire_hand/state/r",
        },
    ),
    "inspire_dfx": (
        "inspire_dfx_state_subscribers",
        {"combined": "rt/inspire/state"},
    ),
}

_DFX_SIDE_COUNTER_KEYS = (
    "drop_event_count",
    "lost_increment_count",
    "reset_count",
    "counter_regression_count",
    "counter_divergence_count",
)

_DFX_REQUIRED_SIDE_COUNTER_KEYS = (
    "sample_count",
    "baseline_count",
    "accepted_count",
    "held_sample_count",
    *_DFX_SIDE_COUNTER_KEYS,
)

_DDS_DEFINITION = (
    "A valid ChannelSubscriber.Read() inter-arrival gap strictly "
    "greater than gap_threshold_s"
)

_DFX_DEFINITION = (
    "Per-side DFX read failures inferred from six identical "
    "MotorState.lost counters; q is held on increments, counter "
    "regressions, divergent counters, and baseline samples"
)

_DFX_DIAGNOSTIC_SOURCE = "inspire_dfx_lost_counters"


def _same_json_contract(actual: Any, expected: Any) -> bool:
    """Compare schema values without Python's bool/int equality shortcut."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_contract(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


@dataclass(frozen=True)
class EpisodeQuality:
    """One episode's compact, JSON-serializable quality result."""

    episode: str
    data_json: str
    end_effector: str
    status: str
    reasons: list[str]
    frame_count: int | None
    measured_fps: float | None
    max_frame_gap_s: float | None
    dds_gap_count: int | None
    dfx_drop_event_count: int | None
    dfx_lost_increment_count: int | None
    dfx_reset_count: int | None
    dfx_divergence_count: int | None
    dfx_malformed_message_count: int | None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _nonnegative_counter(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _walk(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def _is_dds_gap_key(key: str) -> bool:
    return key == "gap_count" or (
        key.startswith("total_") and key.endswith("_gap_count")
    )


def _infer_end_effector(document: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    info = document.get("info")
    contract = info.get("end_effector") if isinstance(info, dict) else None
    if isinstance(contract, dict):
        hand_type = str(contract.get("type", "")).strip().lower().replace("-", "_")
        protocol = str(contract.get("protocol", "")).strip().lower().replace("-", "_")
        if hand_type in {"inspire", "inspire_hand"} and protocol in {"dfx", "ftp"}:
            return f"inspire_{protocol}"
        if hand_type in {"dex3", "unitree_dex3", "unitree_dex3_1"}:
            return "dex3"

    source_names = {str(key).lower().replace("-", "_") for key in diagnostics}
    if any("inspire_dfx" in name for name in source_names):
        return "inspire_dfx"
    if any("inspire_ftp" in name for name in source_names):
        return "inspire_ftp"
    if any("dex3" in name for name in source_names):
        return "dex3"
    return "unknown"


def _validate_end_effector_contract(
    document: dict[str, Any],
    end_effector: str,
) -> list[str]:
    """Require the exact Inspire provenance written by ``EpisodeWriter``."""

    info = document.get("info")
    contract = info.get("end_effector") if isinstance(info, dict) else None
    if end_effector not in {"inspire_dfx", "inspire_ftp"}:
        if contract is not None:
            return ["end_effector_contract_unexpected_or_invalid"]
        return []
    if not isinstance(contract, dict):
        return ["end_effector_contract_missing_or_invalid"]

    expected = dict(_EXPECTED_INSPIRE_END_EFFECTOR_BASE)
    expected["protocol"] = end_effector.removeprefix("inspire_")
    reasons = []
    for key, expected_value in expected.items():
        if not _same_json_contract(contract.get(key), expected_value):
            reasons.append(f"end_effector_contract_mismatch:{key}")

    joint_names = info.get("joint_names")
    if not isinstance(joint_names, dict):
        reasons.append("end_effector_joint_names_missing_or_invalid")
    else:
        for side in ("left", "right"):
            expected_names = expected[f"{side}_joint_names"]
            if not _same_json_contract(joint_names.get(f"{side}_ee"), expected_names):
                reasons.append(f"end_effector_joint_names_mismatch:{side}_ee")
    return reasons


def _validate_frame_evidence(
    frames: Any,
    timing: Any,
) -> tuple[list[str], list[str]]:
    """Cross-check summary timing against the per-frame producer timestamps."""

    unknown: list[str] = []
    failures: list[str] = []
    if not isinstance(frames, list) or not isinstance(timing, dict):
        return unknown, failures

    timestamps: list[float] = []
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            failures.append(f"frame_not_object:index={expected_index}")
            continue
        index = frame.get("idx")
        if isinstance(index, bool) or not isinstance(index, int):
            unknown.append(f"frame_index_missing_or_invalid:index={expected_index}")
        elif index != expected_index:
            failures.append(
                f"frame_index_out_of_sequence:index={expected_index},value={index}"
            )
        timestamp = _finite_number(frame.get("timestamp_s"))
        if timestamp is None:
            unknown.append(f"frame_timestamp_missing_or_invalid:index={expected_index}")
        else:
            timestamps.append(timestamp)

    if len(timestamps) != len(frames):
        return unknown, failures
    if any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        failures.append("frame_timestamps_not_strictly_increasing")
        return unknown, failures

    if len(timestamps) >= 2:
        gaps = [
            current - previous for previous, current in zip(timestamps, timestamps[1:])
        ]
        sample_duration_s = timestamps[-1] - timestamps[0]
        measured_fps = (len(timestamps) - 1) / sample_duration_s
        max_gap_s = max(gaps)
    else:
        sample_duration_s = 0.0
        measured_fps = 0.0
        max_gap_s = 0.0

    derived = {
        "sample_duration_s": sample_duration_s,
        "measured_fps": measured_fps,
        "max_frame_gap_s": max_gap_s,
    }
    for key, expected_value in derived.items():
        recorded = _finite_number(timing.get(key))
        if recorded is None:
            unknown.append(f"{key}_missing_or_invalid")
        elif not math.isclose(recorded, expected_value, rel_tol=1e-9, abs_tol=1e-9):
            failures.append(
                f"{key}_mismatch:timing={recorded:.9f},frames={expected_value:.9f}"
            )

    recording_duration_s = _finite_number(timing.get("recording_duration_s"))
    if recording_duration_s is None:
        unknown.append("recording_duration_s_missing_or_invalid")
    elif timestamps and recording_duration_s + 1e-9 < timestamps[-1]:
        failures.append(
            "recording_duration_shorter_than_last_frame:"
            f"duration={recording_duration_s:.9f},last={timestamps[-1]:.9f}"
        )
    return unknown, failures


def _validate_timing_contract(timing: Any) -> list[str]:
    """Require the non-derived timing provenance emitted by ``EpisodeWriter``."""

    if not isinstance(timing, dict):
        return []
    reasons = []
    target_fps = _finite_number(timing.get("target_fps"))
    if target_fps is None or target_fps <= 0:
        reasons.append("target_fps_missing_or_invalid")
    for key in ("capture_start_utc", "capture_stop_utc"):
        value = timing.get(key)
        if not isinstance(value, str):
            reasons.append(f"{key}_missing_or_invalid")
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            reasons.append(f"{key}_missing_or_invalid")
            continue
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
            reasons.append(f"{key}_missing_or_invalid")
    return reasons


def _validate_dds_event(
    event: Any,
    *,
    source_name: str,
    stream_name: str,
    event_index: int,
    window_duration_s: float,
) -> tuple[list[str], float | None, bool | None]:
    prefix = f"{source_name}.{stream_name}.gaps[{event_index}]"
    reasons = []
    if not isinstance(event, dict):
        return [f"dds_gap_event_missing_or_invalid:{prefix}"], None, None

    start = _finite_number(event.get("start_offset_s"))
    end = _finite_number(event.get("end_offset_s"))
    duration = _finite_number(event.get("duration_s"))
    recovered = event.get("recovered")
    if start is None or end is None or duration is None:
        reasons.append(f"dds_gap_event_timing_missing_or_invalid:{prefix}")
    elif (
        end <= start
        or end > window_duration_s + 1e-9
        or not math.isclose(duration, end - start, rel_tol=1e-9, abs_tol=1e-9)
    ):
        reasons.append(f"dds_gap_event_timing_mismatch:{prefix}")
    if not isinstance(recovered, bool):
        reasons.append(f"dds_gap_event_recovered_missing_or_invalid:{prefix}")
        recovered = None
    return reasons, duration, recovered


def _sum_leaf_counts(
    entries: list[tuple[tuple[str, ...], int]], key: str
) -> int | None:
    values = [value for path, value in entries if path[-1] == key]
    return sum(values) if values else None


def _prefer_total(
    entries: list[tuple[tuple[str, ...], int]],
    total_key: str,
    leaf_key: str,
) -> int | None:
    totals = [value for path, value in entries if path[-1] == total_key]
    if totals:
        # There should be one top-level total.  max() avoids double-counting if
        # a producer also nests a copy of the summary.
        return max(totals)
    return _sum_leaf_counts(entries, leaf_key)


def _validate_dds_source(
    diagnostics: dict[str, Any],
    end_effector: str,
) -> list[str]:
    """Return missing/invalid evidence that prevents a clean classification."""

    contract = _DDS_SOURCE_CONTRACTS.get(end_effector)
    if contract is None:
        return []
    source_name, expected_topics = contract
    source = diagnostics.get(source_name)
    if not isinstance(source, dict):
        return [f"dds_diagnostic_source_missing_or_invalid:{source_name}"]

    reasons = []
    schema_version = source.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        reasons.append(f"dds_schema_version_unsupported:{source_name}")
    if source.get("metric") != "valid_state_receive_gap":
        reasons.append(f"dds_metric_missing_or_invalid:{source_name}")
    if source.get("definition") != _DDS_DEFINITION:
        reasons.append(f"dds_definition_missing_or_invalid:{source_name}")
    gap_threshold_s = _finite_number(source.get("gap_threshold_s"))
    if gap_threshold_s is None or gap_threshold_s <= 0:
        reasons.append(f"dds_gap_threshold_missing_or_invalid:{source_name}")
    window_duration_s = _finite_number(source.get("window_duration_s"))
    if window_duration_s is None:
        reasons.append(f"dds_window_duration_missing_or_invalid:{source_name}")

    expected = set(expected_topics)
    observed = {
        str(key)
        for key, value in source.items()
        if isinstance(value, dict)
        and any(
            field in value
            for field in (
                "gap_count",
                "sample_count",
                "has_received_valid_sample",
                "topic",
            )
        )
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        reasons.append(f"dds_streams_missing:{source_name}:{','.join(missing)}")
    if unexpected:
        reasons.append(f"dds_streams_unexpected:{source_name}:{','.join(unexpected)}")

    stream_gap_counts = []
    for stream_name, expected_topic in expected_topics.items():
        stream = source.get(stream_name)
        if not isinstance(stream, dict):
            continue
        gap_count = _nonnegative_counter(stream.get("gap_count"))
        if gap_count is None:
            reasons.append(
                f"dds_gap_count_missing_or_invalid:{source_name}.{stream_name}"
            )
        else:
            stream_gap_counts.append(gap_count)
        if stream.get("has_received_valid_sample") is not True:
            reasons.append(f"dds_stream_not_observed:{source_name}.{stream_name}")
        sample_count = _nonnegative_counter(stream.get("sample_count"))
        if sample_count is None:
            reasons.append(
                f"dds_sample_count_missing_or_invalid:{source_name}.{stream_name}"
            )
        elif sample_count == 0:
            reasons.append(f"dds_sample_count_zero:{source_name}.{stream_name}")
        topic = stream.get("topic")
        if topic != expected_topic:
            reasons.append(f"dds_topic_missing_or_invalid:{source_name}.{stream_name}")

        recovered_count = _nonnegative_counter(stream.get("recovered_gap_count"))
        if recovered_count is None:
            reasons.append(
                f"dds_recovered_count_missing_or_invalid:{source_name}.{stream_name}"
            )
        open_at_end = stream.get("open_gap_at_end")
        if not isinstance(open_at_end, bool):
            reasons.append(
                f"dds_open_gap_flag_missing_or_invalid:{source_name}.{stream_name}"
            )
        last_sample_age_s = _finite_number(stream.get("last_sample_age_s_at_end"))
        if last_sample_age_s is None:
            reasons.append(
                f"dds_last_sample_age_missing_or_invalid:{source_name}.{stream_name}"
            )
        total_duration_s = _finite_number(stream.get("total_gap_duration_s"))
        max_duration_s = _finite_number(stream.get("max_gap_duration_s"))
        if total_duration_s is None or max_duration_s is None:
            reasons.append(
                f"dds_gap_duration_missing_or_invalid:{source_name}.{stream_name}"
            )

        events = stream.get("gaps")
        if not isinstance(events, list):
            reasons.append(
                f"dds_gap_events_missing_or_invalid:{source_name}.{stream_name}"
            )
            continue
        event_durations = []
        event_recovered = []
        if window_duration_s is not None:
            for event_index, event in enumerate(events):
                event_reasons, duration, recovered = _validate_dds_event(
                    event,
                    source_name=source_name,
                    stream_name=stream_name,
                    event_index=event_index,
                    window_duration_s=window_duration_s,
                )
                reasons.extend(event_reasons)
                if duration is not None:
                    event_durations.append(duration)
                if recovered is not None:
                    event_recovered.append(recovered)
        if gap_count is not None and gap_count != len(events):
            reasons.append(f"dds_gap_event_count_mismatch:{source_name}.{stream_name}")
        if recovered_count is not None and len(event_recovered) == len(events):
            if recovered_count != sum(event_recovered):
                reasons.append(
                    f"dds_recovered_count_mismatch:{source_name}.{stream_name}"
                )
        if isinstance(open_at_end, bool) and len(event_recovered) == len(events):
            if open_at_end != any(not recovered for recovered in event_recovered):
                reasons.append(
                    f"dds_open_gap_flag_mismatch:{source_name}.{stream_name}"
                )
        if len(event_durations) == len(events):
            expected_total = sum(event_durations)
            expected_max = max(event_durations, default=0.0)
            if total_duration_s is not None and not math.isclose(
                total_duration_s, expected_total, rel_tol=1e-9, abs_tol=1e-9
            ):
                reasons.append(
                    f"dds_total_gap_duration_mismatch:{source_name}.{stream_name}"
                )
            if max_duration_s is not None and not math.isclose(
                max_duration_s, expected_max, rel_tol=1e-9, abs_tol=1e-9
            ):
                reasons.append(
                    f"dds_max_gap_duration_mismatch:{source_name}.{stream_name}"
                )
        if (
            gap_count == 0
            and last_sample_age_s is not None
            and gap_threshold_s is not None
            and last_sample_age_s > gap_threshold_s
        ):
            reasons.append(f"dds_open_gap_missing:{source_name}.{stream_name}")

    aggregate_key = (
        "total_stream_gap_count"
        if end_effector == "inspire_dfx"
        else "total_side_gap_count"
    )
    aggregate = _nonnegative_counter(source.get(aggregate_key))
    if aggregate is None:
        reasons.append(
            f"dds_aggregate_missing_or_invalid:{source_name}.{aggregate_key}"
        )
    elif len(stream_gap_counts) == len(expected_topics) and aggregate != sum(
        stream_gap_counts
    ):
        reasons.append(f"dds_aggregate_mismatch:{source_name}.{aggregate_key}")
    any_gap = source.get("any_gap")
    if not isinstance(any_gap, bool):
        reasons.append(f"dds_any_gap_missing_or_invalid:{source_name}")
    elif aggregate is not None and any_gap != (aggregate > 0):
        reasons.append(f"dds_any_gap_mismatch:{source_name}")
    return reasons


def _validate_dfx_source(diagnostics: dict[str, Any]) -> list[str]:
    """Require the complete DFX per-side lost-counter evidence."""

    source_name = _DFX_DIAGNOSTIC_SOURCE
    source = diagnostics.get(source_name)
    if not isinstance(source, dict):
        return [f"dfx_diagnostic_source_missing_or_invalid:{source_name}"]

    reasons = []
    schema_version = source.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        reasons.append(f"dfx_schema_version_unsupported:{source_name}")
    if source.get("metric") != "inspire_dfx_motor_state_lost_counter":
        reasons.append(f"dfx_metric_missing_or_invalid:{source_name}")
    if source.get("definition") != _DFX_DEFINITION:
        reasons.append(f"dfx_definition_missing_or_invalid:{source_name}")
    if _finite_number(source.get("window_duration_s")) is None:
        reasons.append(f"dfx_window_duration_missing_or_invalid:{source_name}")

    malformed_count = _nonnegative_counter(source.get("malformed_message_count"))
    malformed_messages = source.get("malformed_messages")
    if not isinstance(malformed_messages, list):
        reasons.append("dfx_malformed_messages_missing_or_invalid")
    elif malformed_count is not None and len(malformed_messages) != malformed_count:
        reasons.append("dfx_malformed_message_accounting_mismatch")

    observed_sides = {
        str(key)
        for key, value in source.items()
        if isinstance(value, dict)
        and any(counter in value for counter in _DFX_SIDE_COUNTER_KEYS)
    }
    expected_sides = {"left", "right"}
    missing_sides = sorted(expected_sides - observed_sides)
    unexpected_sides = sorted(observed_sides - expected_sides)
    if missing_sides:
        reasons.append(f"dfx_sides_missing:{','.join(missing_sides)}")
    if unexpected_sides:
        reasons.append(f"dfx_sides_unexpected:{','.join(unexpected_sides)}")

    for side_name in ("left", "right"):
        side = source.get(side_name)
        if not isinstance(side, dict):
            continue
        values = {}
        for counter_name in _DFX_REQUIRED_SIDE_COUNTER_KEYS:
            value = _nonnegative_counter(side.get(counter_name))
            if value is None:
                reasons.append(
                    f"dfx_counter_missing_or_invalid:{side_name}.{counter_name}"
                )
            else:
                values[counter_name] = value
        if len(values) == len(_DFX_REQUIRED_SIDE_COUNTER_KEYS):
            if values["sample_count"] == 0:
                reasons.append(f"dfx_sample_count_zero:{side_name}")
            if values["accepted_count"] == 0:
                reasons.append(f"dfx_accepted_count_zero:{side_name}")
            if (
                values["sample_count"]
                != values["accepted_count"] + values["held_sample_count"]
            ):
                reasons.append(f"dfx_sample_accounting_mismatch:{side_name}")
            if values["held_sample_count"] != (
                values["baseline_count"]
                + values["drop_event_count"]
                + values["reset_count"]
            ):
                reasons.append(f"dfx_held_accounting_mismatch:{side_name}")
            if values["reset_count"] != (
                values["counter_regression_count"] + values["counter_divergence_count"]
            ):
                reasons.append(f"dfx_reset_accounting_mismatch:{side_name}")

        events = side.get("events")
        if not isinstance(events, list):
            reasons.append(f"dfx_events_missing_or_invalid:{side_name}")
        elif (
            "held_sample_count" in values and len(events) != values["held_sample_count"]
        ):
            reasons.append(f"dfx_event_accounting_mismatch:{side_name}")

        for field in ("baseline_at_start", "baseline_at_end", "last_observed_counters"):
            if field not in side:
                reasons.append(
                    f"dfx_state_counters_missing_or_invalid:{side_name}.{field}"
                )
                continue
            counters = side.get(field)
            if counters is not None and (
                not isinstance(counters, list)
                or len(counters) != 6
                or any(_nonnegative_counter(value) is None for value in counters)
            ):
                reasons.append(
                    f"dfx_state_counters_missing_or_invalid:{side_name}.{field}"
                )
        if values.get("accepted_count", 0) > 0 and (
            side.get("baseline_at_end") is None
            or side.get("last_observed_counters") is None
        ):
            reasons.append(f"dfx_state_counters_missing_or_invalid:{side_name}")

    for counter_name in (
        "malformed_message_count",
        "total_drop_event_count",
        "total_held_sample_count",
        "total_lost_increment_count",
        "total_reset_count",
    ):
        if _nonnegative_counter(source.get(counter_name)) is None:
            reasons.append(f"dfx_counter_missing_or_invalid:{counter_name}")

    side_values = [source.get(side) for side in ("left", "right")]
    if all(isinstance(side, dict) for side in side_values):
        for total_key, side_key in (
            ("total_drop_event_count", "drop_event_count"),
            ("total_held_sample_count", "held_sample_count"),
            ("total_lost_increment_count", "lost_increment_count"),
            ("total_reset_count", "reset_count"),
        ):
            total = _nonnegative_counter(source.get(total_key))
            parts = [_nonnegative_counter(side.get(side_key)) for side in side_values]
            if total is not None and all(part is not None for part in parts):
                if total != sum(parts):
                    reasons.append(f"dfx_total_accounting_mismatch:{total_key}")
    any_state_held = source.get("any_state_held")
    any_anomaly = source.get("any_anomaly")
    if not isinstance(any_state_held, bool):
        reasons.append("dfx_flag_missing_or_invalid:any_state_held")
    else:
        total_held = _nonnegative_counter(source.get("total_held_sample_count"))
        if total_held is not None and any_state_held != (total_held > 0):
            reasons.append("dfx_flag_mismatch:any_state_held")
    if not isinstance(any_anomaly, bool):
        reasons.append("dfx_flag_missing_or_invalid:any_anomaly")
    else:
        anomaly_parts = [
            _nonnegative_counter(source.get("total_drop_event_count")),
            _nonnegative_counter(source.get("total_reset_count")),
            malformed_count,
        ]
        if all(value is not None for value in anomaly_parts) and any_anomaly != (
            sum(anomaly_parts) > 0
        ):
            reasons.append("dfx_flag_mismatch:any_anomaly")

    if all(isinstance(side, dict) for side in side_values):
        sample_counts = [
            _nonnegative_counter(side.get("sample_count")) for side in side_values
        ]
        if (
            all(value is not None for value in sample_counts)
            and len(set(sample_counts)) != 1
        ):
            reasons.append("dfx_side_sample_count_mismatch")
    return reasons


def _empty_result(path: Path, episode: str, reason: str) -> EpisodeQuality:
    return EpisodeQuality(
        episode=episode,
        data_json=str(path.resolve()),
        end_effector="unknown",
        status="unknown",
        reasons=[reason],
        frame_count=None,
        measured_fps=None,
        max_frame_gap_s=None,
        dds_gap_count=None,
        dfx_drop_event_count=None,
        dfx_lost_increment_count=None,
        dfx_reset_count=None,
        dfx_divergence_count=None,
        dfx_malformed_message_count=None,
    )


def classify_episode(
    data_json: Path,
    *,
    display_name: str | None = None,
    max_frame_gap_s: float = DEFAULT_MAX_FRAME_GAP_S,
    min_measured_fps: float = DEFAULT_MIN_MEASURED_FPS,
) -> EpisodeQuality:
    """Classify one ``data.json`` without writing to it or its directory."""

    if not math.isfinite(max_frame_gap_s) or max_frame_gap_s < 0:
        raise ValueError("max_frame_gap_s must be finite and non-negative")
    if not math.isfinite(min_measured_fps) or min_measured_fps < 0:
        raise ValueError("min_measured_fps must be finite and non-negative")

    path = Path(data_json)
    episode = display_name or path.parent.name
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return _empty_result(
            path, episode, f"unreadable_data_json:{type(error).__name__}"
        )
    if not isinstance(document, dict):
        return _empty_result(path, episode, "data_json_root_not_object")

    timing = document.get("timing")
    diagnostics = document.get("diagnostics")
    hard_failures: list[str] = []
    unknown_reasons: list[str] = []

    frame_count: int | None = None
    measured_fps: float | None = None
    observed_max_gap: float | None = None
    if isinstance(timing, dict):
        raw_frame_count = timing.get("frame_count")
        if (
            isinstance(raw_frame_count, int)
            and not isinstance(raw_frame_count, bool)
            and raw_frame_count >= 0
        ):
            frame_count = raw_frame_count
        measured_fps = _finite_number(timing.get("measured_fps"))
        observed_max_gap = _finite_number(timing.get("max_frame_gap_s"))
    else:
        unknown_reasons.append("timing_missing")

    frames = document.get("data")
    if not isinstance(frames, list):
        unknown_reasons.append("data_frames_missing_or_invalid")
    if frame_count is None:
        unknown_reasons.append("frame_count_missing_or_invalid")
    elif isinstance(frames, list) and frame_count != len(frames):
        hard_failures.append(
            f"frame_count_mismatch:timing={frame_count},data={len(frames)}"
        )
    frame_unknown, frame_failures = _validate_frame_evidence(frames, timing)
    unknown_reasons.extend(frame_unknown)
    hard_failures.extend(frame_failures)
    unknown_reasons.extend(_validate_timing_contract(timing))

    if measured_fps is None:
        unknown_reasons.append("measured_fps_missing_or_invalid")
    elif measured_fps < min_measured_fps:
        hard_failures.append(f"measured_fps={measured_fps:.3f}<{min_measured_fps:.3f}")
    if observed_max_gap is None:
        unknown_reasons.append("max_frame_gap_missing_or_invalid")
    elif observed_max_gap > max_frame_gap_s:
        hard_failures.append(
            f"max_frame_gap_s={observed_max_gap:.6f}>{max_frame_gap_s:.6f}"
        )

    if not isinstance(diagnostics, dict) or not diagnostics:
        end_effector = _infer_end_effector(document, {})
        unknown_reasons.append("diagnostics_missing")
        counter_entries: list[tuple[tuple[str, ...], int]] = []
        dds_entries: list[tuple[tuple[str, ...], int]] = []
    else:
        end_effector = _infer_end_effector(document, diagnostics)
        counter_entries = []
        dds_entries = []
        invalid_counter_paths: list[str] = []
        diagnostics_unavailable = False
        expected_dds = _DDS_SOURCE_CONTRACTS.get(end_effector)
        relevant_sources = set()
        if expected_dds is not None:
            relevant_sources.add(expected_dds[0])
        if end_effector == "inspire_dfx":
            relevant_sources.add(_DFX_DIAGNOSTIC_SOURCE)
        for source_name in relevant_sources:
            source = diagnostics.get(source_name)
            if not isinstance(source, dict):
                continue
            for path_parts, value in _walk(source, (source_name,)):
                key = path_parts[-1]
                if key == "available" and value is False:
                    diagnostics_unavailable = True
                is_expected_dds_counter = (
                    expected_dds is not None
                    and source_name == expected_dds[0]
                    and _is_dds_gap_key(key)
                )
                is_expected_dfx_counter = (
                    source_name == _DFX_DIAGNOSTIC_SOURCE and key in _DFX_COUNTER_KEYS
                )
                if is_expected_dds_counter or is_expected_dfx_counter:
                    count = _nonnegative_counter(value)
                    if count is None:
                        invalid_counter_paths.append(".".join(path_parts))
                        continue
                    counter_entries.append((path_parts, count))
                    if is_expected_dds_counter:
                        dds_entries.append((path_parts, count))

        if diagnostics_unavailable:
            unknown_reasons.append("diagnostics_unavailable")
        unknown_reasons.extend(
            f"diagnostic_counter_invalid:{path}" for path in invalid_counter_paths
        )
        if not dds_entries:
            unknown_reasons.append("dds_gap_metrics_missing")
        unknown_reasons.extend(_validate_dds_source(diagnostics, end_effector))
        if end_effector == "inspire_dfx":
            unknown_reasons.extend(_validate_dfx_source(diagnostics))

        recognized_sources = {
            source_name
            for source_name, _ in _DDS_SOURCE_CONTRACTS.values()
            if source_name in diagnostics
        }
        if _DFX_DIAGNOSTIC_SOURCE in diagnostics:
            recognized_sources.add(_DFX_DIAGNOSTIC_SOURCE)
        if recognized_sources != relevant_sources:
            unknown_reasons.append(
                "hand_diagnostic_source_set_mismatch:expected="
                + ",".join(sorted(relevant_sources))
                + ":observed="
                + ",".join(sorted(recognized_sources))
            )

    unknown_reasons.extend(_validate_end_effector_contract(document, end_effector))

    # Sum only stream-level gap_count values for display; aggregate keys are
    # still checked below, so an inconsistent non-zero total cannot pass.
    dds_gap_count = _sum_leaf_counts(dds_entries, "gap_count")
    if dds_gap_count is None and dds_entries:
        dds_gap_count = max(count for _, count in dds_entries)
    for counter_path, count in dds_entries:
        if count > 0:
            hard_failures.append(f"dds_gap:{'.'.join(counter_path)}={count}")

    is_dfx = end_effector == "inspire_dfx"
    dfx_drop_events = _prefer_total(
        counter_entries, "total_drop_event_count", "drop_event_count"
    )
    dfx_lost_increments = _prefer_total(
        counter_entries, "total_lost_increment_count", "lost_increment_count"
    )
    dfx_resets = _prefer_total(counter_entries, "total_reset_count", "reset_count")
    dfx_divergence = _sum_leaf_counts(counter_entries, "counter_divergence_count")
    dfx_malformed = _sum_leaf_counts(counter_entries, "malformed_message_count")

    if is_dfx:
        required_dfx_metrics = {
            "drop_event": dfx_drop_events,
            "lost_increment": dfx_lost_increments,
            "reset": dfx_resets,
            "divergence": dfx_divergence,
            "malformed": dfx_malformed,
        }
        missing = [
            name for name, value in required_dfx_metrics.items() if value is None
        ]
        if missing:
            unknown_reasons.append("dfx_metrics_missing:" + ",".join(missing))
        for counter_path, count in counter_entries:
            if counter_path[-1] in _DFX_COUNTER_KEYS and count > 0:
                hard_failures.append(f"dfx:{'.'.join(counter_path)}={count}")
    else:
        dfx_drop_events = None
        dfx_lost_increments = None
        dfx_resets = None
        dfx_divergence = None
        dfx_malformed = None

    if end_effector == "unknown":
        unknown_reasons.append("end_effector_unknown")

    # Reject wins when there is direct evidence of an anomaly.  Otherwise,
    # missing evidence remains unknown rather than being treated as clean.
    if hard_failures:
        status = "reject"
        reasons = hard_failures + unknown_reasons
    elif unknown_reasons:
        status = "unknown"
        reasons = unknown_reasons
    else:
        status = "clean"
        reasons = ["all_required_metrics_within_limits"]

    return EpisodeQuality(
        episode=episode,
        data_json=str(path.resolve()),
        end_effector=end_effector,
        status=status,
        reasons=reasons,
        frame_count=frame_count,
        measured_fps=measured_fps,
        max_frame_gap_s=observed_max_gap,
        dds_gap_count=dds_gap_count,
        dfx_drop_event_count=dfx_drop_events,
        dfx_lost_increment_count=dfx_lost_increments,
        dfx_reset_count=dfx_resets,
        dfx_divergence_count=dfx_divergence,
        dfx_malformed_message_count=dfx_malformed,
    )


def discover_episode_files(task_dir: Path) -> list[Path]:
    """Find raw episode JSON files under a task directory, deterministically."""

    root = Path(task_dir)
    if root.is_file():
        return [root]
    direct = root / "data.json"
    if direct.is_file():
        return [direct]
    if not root.is_dir():
        raise FileNotFoundError(f"task directory does not exist: {root}")
    return sorted(root.glob("episode_*/data.json"), key=lambda item: str(item.parent))


def scan_task(
    task_dir: Path,
    *,
    max_frame_gap_s: float = DEFAULT_MAX_FRAME_GAP_S,
    min_measured_fps: float = DEFAULT_MIN_MEASURED_FPS,
) -> list[EpisodeQuality]:
    root = Path(task_dir)
    files = discover_episode_files(root)
    results = []
    for path in files:
        try:
            display_name = str(path.parent.relative_to(root))
        except ValueError:
            display_name = path.parent.name
        if display_name == ".":
            display_name = path.parent.name
        results.append(
            classify_episode(
                path,
                display_name=display_name,
                max_frame_gap_s=max_frame_gap_s,
                min_measured_fps=min_measured_fps,
            )
        )
    return results


def _display(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def render_table(results: list[EpisodeQuality]) -> str:
    headers = (
        "EPISODE",
        "HAND",
        "FRAMES",
        "FPS",
        "FRAME_GAP",
        "DDS_GAPS",
        "DFX_DROP",
        "DFX_LOST",
        "RESET",
        "DIVERGE",
        "MALFORM",
        "STATUS",
        "REASON",
    )
    rows = []
    for result in results:
        rows.append(
            (
                result.episode,
                result.end_effector,
                _display(result.frame_count),
                _display(result.measured_fps),
                _display(result.max_frame_gap_s, 6),
                _display(result.dds_gap_count),
                _display(result.dfx_drop_event_count),
                _display(result.dfx_lost_increment_count),
                _display(result.dfx_reset_count),
                _display(result.dfx_divergence_count),
                _display(result.dfx_malformed_message_count),
                result.status,
                ";".join(result.reasons),
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    ]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    counts = Counter(result.status for result in results)
    lines.append(
        f"TOTAL {len(results)}  CLEAN {counts['clean']}  REJECT {counts['reject']}  "
        f"UNKNOWN {counts['unknown']}"
    )
    return "\n".join(lines)


def build_manifest(
    task_dir: Path,
    results: list[EpisodeQuality],
    *,
    max_frame_gap_s: float,
    min_measured_fps: float,
) -> dict[str, Any]:
    counts = Counter(result.status for result in results)
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task_dir": str(Path(task_dir).resolve()),
        "criteria": {
            "max_frame_gap_s": max_frame_gap_s,
            "min_measured_fps": min_measured_fps,
            "require_frame_count_matches_data": True,
            "require_dds_gap_diagnostics": True,
            "required_dds_streams": {
                end_effector: list(streams)
                for end_effector, (_, streams) in _DDS_SOURCE_CONTRACTS.items()
            },
            "require_dfx_lost_counter_diagnostics_for_inspire_dfx": True,
            "legacy_without_diagnostics": "unknown",
        },
        "counts": {
            "clean": counts["clean"],
            "reject": counts["reject"],
            "unknown": counts["unknown"],
        },
        "clean": [result.episode for result in results if result.status == "clean"],
        "reject": [result.episode for result in results if result.status == "reject"],
        "unknown": [result.episode for result in results if result.status == "unknown"],
        "episodes": [asdict(result) for result in results],
    }


def _write_exclusive(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        output.write(content)


def _manifest_tsv(results: list[EpisodeQuality]) -> str:
    output = io.StringIO(newline="")
    fieldnames = (
        list(asdict(results[0]).keys())
        if results
        else [field.name for field in EpisodeQuality.__dataclass_fields__.values()]
    )
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for result in results:
        row = asdict(result)
        row["reasons"] = ";".join(result.reasons)
        writer.writerow(row)
    return output.getvalue()


def _nonnegative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read raw teleop episode data.json files and report timing/DDS quality. "
            "Episodes are never changed, deleted, or moved."
        )
    )
    parser.add_argument(
        "task_dir",
        type=Path,
        help="Task directory containing episode_*/data.json (or one episode/data.json).",
    )
    parser.add_argument(
        "--max-frame-gap-s",
        type=_nonnegative_float,
        default=DEFAULT_MAX_FRAME_GAP_S,
        help=f"Reject above this recorded frame gap (default: {DEFAULT_MAX_FRAME_GAP_S}).",
    )
    parser.add_argument(
        "--min-measured-fps",
        type=_nonnegative_float,
        default=DEFAULT_MIN_MEASURED_FPS,
        help=f"Reject below this measured frame rate (default: {DEFAULT_MIN_MEASURED_FPS}).",
    )
    parser.add_argument(
        "--json-manifest",
        type=Path,
        help="Create a JSON clean/reject/unknown manifest; refuses to overwrite.",
    )
    parser.add_argument(
        "--tsv-manifest",
        type=Path,
        help="Create a TSV clean/reject/unknown manifest; refuses to overwrite.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.json_manifest is not None
        and args.tsv_manifest is not None
        and args.json_manifest.resolve() == args.tsv_manifest.resolve()
    ):
        print("error: JSON and TSV manifest paths must differ", file=sys.stderr)
        return 2
    try:
        results = scan_task(
            args.task_dir,
            max_frame_gap_s=args.max_frame_gap_s,
            min_measured_fps=args.min_measured_fps,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not results:
        print(
            f"error: no episode_*/data.json files found under {args.task_dir}",
            file=sys.stderr,
        )
        return 2

    print(render_table(results))
    manifest = build_manifest(
        args.task_dir,
        results,
        max_frame_gap_s=args.max_frame_gap_s,
        min_measured_fps=args.min_measured_fps,
    )
    try:
        manifest_paths = [
            path for path in (args.json_manifest, args.tsv_manifest) if path is not None
        ]
        existing_paths = [path for path in manifest_paths if path.exists()]
        if existing_paths:
            print(
                "error: manifest already exists; refusing to overwrite: "
                + ", ".join(str(path) for path in existing_paths),
                file=sys.stderr,
            )
            return 2
        if args.json_manifest is not None:
            _write_exclusive(
                args.json_manifest,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
        if args.tsv_manifest is not None:
            _write_exclusive(args.tsv_manifest, _manifest_tsv(results))
    except FileExistsError as error:
        print(
            f"error: manifest already exists; refusing to overwrite: {error.filename}",
            file=sys.stderr,
        )
        return 2
    except OSError as error:
        print(f"error: could not write manifest: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
