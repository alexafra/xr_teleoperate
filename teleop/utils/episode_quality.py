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
    "drop_event_count",
    "lost_increment_count",
    "reset_count",
    "counter_regression_count",
    "counter_divergence_count",
    "malformed_message_count",
    "total_drop_event_count",
    "total_lost_increment_count",
    "total_reset_count",
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

    for stream_name, expected_topic in expected_topics.items():
        stream = source.get(stream_name)
        if not isinstance(stream, dict):
            continue
        if _nonnegative_counter(stream.get("gap_count")) is None:
            reasons.append(
                f"dds_gap_count_missing_or_invalid:{source_name}.{stream_name}"
            )
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
    return reasons


def _validate_dfx_source(diagnostics: dict[str, Any]) -> list[str]:
    """Require the complete DFX per-side lost-counter evidence."""

    source_name = "inspire_dfx_lost_counters"
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
        for counter_name in _DFX_SIDE_COUNTER_KEYS:
            if _nonnegative_counter(side.get(counter_name)) is None:
                reasons.append(
                    f"dfx_counter_missing_or_invalid:{side_name}.{counter_name}"
                )

    for counter_name in (
        "malformed_message_count",
        "total_drop_event_count",
        "total_lost_increment_count",
        "total_reset_count",
    ):
        if _nonnegative_counter(source.get(counter_name)) is None:
            reasons.append(f"dfx_counter_missing_or_invalid:{counter_name}")
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
        for path_parts, value in _walk(diagnostics):
            key = path_parts[-1]
            if key == "available" and value is False:
                diagnostics_unavailable = True
            if _is_dds_gap_key(key) or key in _DFX_COUNTER_KEYS:
                count = _nonnegative_counter(value)
                if count is None:
                    invalid_counter_paths.append(".".join(path_parts))
                    continue
                counter_entries.append((path_parts, count))
                if _is_dds_gap_key(key):
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

    # Sum only stream-level gap_count values for display; aggregate keys are
    # still checked below, so an inconsistent non-zero total cannot pass.
    dds_gap_count = _sum_leaf_counts(dds_entries, "gap_count")
    if dds_gap_count is None and dds_entries:
        dds_gap_count = max(count for _, count in dds_entries)
    for counter_path, count in dds_entries:
        if count > 0:
            hard_failures.append(f"dds_gap:{'.'.join(counter_path)}={count}")

    is_dfx = end_effector == "inspire_dfx" or any(
        "inspire_dfx" in ".".join(path_parts).lower()
        for path_parts, _ in counter_entries
    )
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
