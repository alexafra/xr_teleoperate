"""Hardware-independent receive-gap diagnostics for DDS subscriber streams."""

import threading
import time


class SubscriberDropTracker:
    """Track valid-message receive gaps inside explicit recording windows."""

    def __init__(
        self,
        stream_topics,
        gap_threshold_s,
        clock_ns=None,
        *,
        stream_label="subscriber stream",
        window_label="Subscriber diagnostics",
        aggregate_count_key="total_stream_gap_count",
    ):
        if gap_threshold_s <= 0:
            raise ValueError("gap_threshold_s must be positive")
        if not stream_topics:
            raise ValueError("stream_topics must not be empty")

        self.gap_threshold_s = float(gap_threshold_s)
        self._gap_threshold_ns = int(round(self.gap_threshold_s * 1e9))
        self._clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
        self._stream_topics = dict(stream_topics)
        self._streams = tuple(self._stream_topics)
        self._stream_label = stream_label
        self._window_label = window_label
        self._aggregate_count_key = aggregate_count_key
        self._lock = threading.Lock()
        self._last_valid_ns = {stream: None for stream in self._streams}
        self._window_start_ns = None
        self._window_events = {stream: [] for stream in self._streams}
        self._window_sample_counts = {stream: 0 for stream in self._streams}

    def observe(self, stream, received_ns=None):
        if stream not in self._stream_topics:
            raise ValueError(f"Unknown {self._stream_label}: {stream}")

        now_ns = self._clock_ns() if received_ns is None else int(received_ns)
        with self._lock:
            previous_ns = self._last_valid_ns[stream]
            self._last_valid_ns[stream] = now_ns

            if self._window_start_ns is None or now_ns < self._window_start_ns:
                return

            self._window_sample_counts[stream] += 1
            if (
                previous_ns is not None
                and now_ns > previous_ns
                and now_ns - previous_ns > self._gap_threshold_ns
            ):
                self._window_events[stream].append((previous_ns, now_ns, True))

    def begin_episode_window(self):
        with self._lock:
            if self._window_start_ns is not None:
                raise RuntimeError(f"{self._window_label} window is already active")
            now_ns = self._clock_ns()
            self._window_start_ns = now_ns
            self._window_events = {stream: [] for stream in self._streams}
            self._window_sample_counts = {stream: 0 for stream in self._streams}

    def finish_episode_window(self):
        with self._lock:
            start_ns = self._window_start_ns
            if start_ns is None:
                raise RuntimeError(f"{self._window_label} window is not active")
            end_ns = self._clock_ns()
            if end_ns < start_ns:
                raise RuntimeError(
                    f"{self._window_label} clock moved backwards"
                )

            window_events = {
                stream: list(self._window_events[stream])
                for stream in self._streams
            }
            window_sample_counts = dict(self._window_sample_counts)
            last_valid_ns_by_stream = dict(self._last_valid_ns)
            self._window_start_ns = None
            self._window_events = {stream: [] for stream in self._streams}
            self._window_sample_counts = {stream: 0 for stream in self._streams}

        window_duration_s = (end_ns - start_ns) / 1e9
        summary = {
            "schema_version": 1,
            "metric": "valid_state_receive_gap",
            "definition": (
                "A valid ChannelSubscriber.Read() inter-arrival gap strictly "
                "greater than gap_threshold_s"
            ),
            "gap_threshold_s": self.gap_threshold_s,
            "window_duration_s": window_duration_s,
        }

        aggregate_gap_count = 0
        for stream in self._streams:
            events = window_events[stream]
            last_valid_ns = last_valid_ns_by_stream[stream]
            last_sample_age_s = None

            if last_valid_ns is not None:
                last_sample_age_ns = max(0, end_ns - last_valid_ns)
                last_sample_age_s = last_sample_age_ns / 1e9
                if last_sample_age_ns > self._gap_threshold_ns:
                    events.append((last_valid_ns, end_ns, False))

            event_rows = []
            for gap_start_ns, gap_end_ns, recovered in events:
                clipped_start_ns = max(gap_start_ns, start_ns)
                clipped_end_ns = min(gap_end_ns, end_ns)
                if clipped_end_ns <= clipped_start_ns:
                    continue
                event_rows.append(
                    {
                        "start_offset_s": (clipped_start_ns - start_ns) / 1e9,
                        "end_offset_s": (clipped_end_ns - start_ns) / 1e9,
                        "duration_s": (clipped_end_ns - clipped_start_ns) / 1e9,
                        "recovered": bool(recovered),
                    }
                )

            gap_durations = [event["duration_s"] for event in event_rows]
            recovered_count = sum(event["recovered"] for event in event_rows)
            open_at_end = any(not event["recovered"] for event in event_rows)
            stream_gap_count = len(event_rows)
            aggregate_gap_count += stream_gap_count
            summary[stream] = {
                "topic": self._stream_topics[stream],
                "has_received_valid_sample": last_valid_ns is not None,
                "sample_count": int(window_sample_counts[stream]),
                "gap_count": stream_gap_count,
                "recovered_gap_count": recovered_count,
                "open_gap_at_end": open_at_end,
                "last_sample_age_s_at_end": last_sample_age_s,
                "total_gap_duration_s": sum(gap_durations),
                "max_gap_duration_s": max(gap_durations, default=0.0),
                "gaps": event_rows,
            }

        summary[self._aggregate_count_key] = aggregate_gap_count
        summary["any_gap"] = aggregate_gap_count > 0
        return summary
