#!/usr/bin/env python3
"""
Benchmark the real xr_teleoperate EpisodeWriter without operating the robot.

Run this file from:
    /home/alex/xr_teleoperate/teleop

The benchmark receives three ZMQ streams:

    RGB:               port 5555
    Aligned depth:     port 5558
    Raw depth:         port 5559

It decodes all three streams and submits items at the requested frequency
to the project's installed EpisodeWriter.

Expected episode output:

    colors/
    depths/
    raw_depths/
    data.json

By default Rerun is disabled, so the first test measures:

    network
    -> RGB/depth decoding
    -> EpisodeWriter queue
    -> JPEG/PNG writing
    -> JSON writing
    -> disk

Use --rerun for a second comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleop.utils.episode_writer import EpisodeWriter  # noqa: E402


def create_subscriber(
    context: zmq.Context,
    host: str,
    port: int,
) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(f"tcp://{host}:{port}")
    return socket


def decode(
    encoded: bytes,
    mode: int,
) -> np.ndarray | None:
    return cv2.imdecode(
        np.frombuffer(
            encoded,
            dtype=np.uint8,
        ),
        mode,
    )


def synthetic_robot_data() -> tuple[dict, dict]:
    """
    Match the broad state/action structure used by teleop_hand_and_arm.py.

    Values are synthetic and exist only to approximate the JSON and Rerun
    workload without operating the robot.
    """
    states = {
        "left_arm": {
            "qpos": [0.0] * 7,
            "qvel": [0.0] * 7,
            "torque": [],
        },
        "right_arm": {
            "qpos": [0.0] * 7,
            "qvel": [0.0] * 7,
            "torque": [],
        },
        "left_ee": {
            "qpos": [0.0] * 7,
            "qvel": [],
            "torque": [],
        },
        "right_ee": {
            "qpos": [0.0] * 7,
            "qvel": [],
            "torque": [],
        },
        "body": {
            "qpos": [],
        },
    }

    actions = {
        "left_arm": {
            "qpos": [0.0] * 7,
            "qvel": [],
            "torque": [],
        },
        "right_arm": {
            "qpos": [0.0] * 7,
            "qvel": [],
            "torque": [],
        },
        "left_ee": {
            "qpos": [0.0] * 7,
            "qvel": [],
            "torque": [],
        },
        "right_ee": {
            "qpos": [0.0] * 7,
            "qvel": [],
            "torque": [],
        },
        "body": {
            "qpos": [],
        },
    }

    return states, actions


def directory_size(path: Path) -> int:
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark xr_teleoperate EpisodeWriter with RGB, "
            "aligned depth and raw depth."
        )
    )

    parser.add_argument(
        "--host",
        default="192.168.123.164",
    )

    parser.add_argument(
        "--rgb-port",
        type=int,
        default=5555,
    )

    parser.add_argument(
        "--depth-port",
        type=int,
        default=5558,
    )

    parser.add_argument(
        "--raw-depth-port",
        type=int,
        default=5559,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--frequency",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--task-dir",
        default="./utils/data/episode_writer_benchmark",
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Enable Rerun for a comparison test.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context = zmq.Context()

    rgb_socket = create_subscriber(
        context,
        args.host,
        args.rgb_port,
    )

    depth_socket = create_subscriber(
        context,
        args.host,
        args.depth_port,
    )

    raw_depth_socket = create_subscriber(
        context,
        args.host,
        args.raw_depth_port,
    )

    poller = zmq.Poller()

    poller.register(
        rgb_socket,
        zmq.POLLIN,
    )

    poller.register(
        depth_socket,
        zmq.POLLIN,
    )

    poller.register(
        raw_depth_socket,
        zmq.POLLIN,
    )

    writer = EpisodeWriter(
        task_dir=args.task_dir,
        task_goal="EpisodeWriter RGB-D and raw-depth benchmark",
        task_desc=(
            "Synthetic writer-throughput test using RGB, aligned depth "
            "and raw depth; no robot operation."
        ),
        task_steps=(
            "Receive all three image streams and write them at the "
            "requested frequency."
        ),
        frequency=args.frequency,
        rerun_log=args.rerun,
    )

    if not writer.create_episode():
        raise RuntimeError(
            "EpisodeWriter could not create a benchmark episode."
        )

    states, actions = synthetic_robot_data()

    latest_rgb = None
    latest_depth = None
    latest_raw_depth = None

    print(f"RGB:       tcp://{args.host}:{args.rgb_port}")
    print(f"Depth:     tcp://{args.host}:{args.depth_port}")
    print(
        f"Raw depth: tcp://{args.host}:{args.raw_depth_port}"
    )
    print(f"Frequency: {args.frequency:.1f} FPS")
    print(f"Duration:  {args.duration:.1f} seconds")
    print(
        f"Rerun:     "
        f"{'enabled' if args.rerun else 'disabled'}"
    )
    print(f"Output:    {writer.episode_dir}")
    print(
        f"Warming up for {args.warmup:.1f} seconds..."
    )

    warmup_deadline = (
        time.perf_counter() + args.warmup
    )

    while time.perf_counter() < warmup_deadline:
        events = dict(
            poller.poll(timeout=100)
        )

        if rgb_socket in events:
            latest_rgb = decode(
                rgb_socket.recv(),
                cv2.IMREAD_COLOR,
            )

        if depth_socket in events:
            latest_depth = decode(
                depth_socket.recv(),
                cv2.IMREAD_UNCHANGED,
            )

        if raw_depth_socket in events:
            latest_raw_depth = decode(
                raw_depth_socket.recv(),
                cv2.IMREAD_UNCHANGED,
            )

    missing_streams = []

    if latest_rgb is None:
        missing_streams.append(
            f"RGB port {args.rgb_port}"
        )

    if latest_depth is None:
        missing_streams.append(
            f"aligned-depth port {args.depth_port}"
        )

    if latest_raw_depth is None:
        missing_streams.append(
            f"raw-depth port {args.raw_depth_port}"
        )

    if missing_streams:
        writer.save_episode()

        while not writer.is_ready():
            time.sleep(0.01)

        writer.close()

        rgb_socket.close()
        depth_socket.close()
        raw_depth_socket.close()
        context.term()

        raise RuntimeError(
            "No frame received during warm-up from: "
            + ", ".join(missing_streams)
        )

    if latest_depth.dtype != np.uint16:
        raise RuntimeError(
            "Aligned depth should decode as uint16, "
            f"but received {latest_depth.dtype}."
        )

    if latest_raw_depth.dtype != np.uint16:
        raise RuntimeError(
            "Raw depth should decode as uint16, "
            f"but received {latest_raw_depth.dtype}."
        )

    print(
        f"RGB ready:       shape={latest_rgb.shape}, "
        f"dtype={latest_rgb.dtype}"
    )

    print(
        f"Depth ready:     shape={latest_depth.shape}, "
        f"dtype={latest_depth.dtype}"
    )

    print(
        f"Raw depth ready: shape={latest_raw_depth.shape}, "
        f"dtype={latest_raw_depth.dtype}"
    )

    period = 1.0 / args.frequency
    started = time.perf_counter()
    deadline = started + args.duration
    next_item_time = started
    next_report_time = started + 1.0

    enqueued = 0
    maximum_backlog = 0

    try:
        while time.perf_counter() < deadline:
            events = dict(
                poller.poll(timeout=1)
            )

            if rgb_socket in events:
                decoded_rgb = decode(
                    rgb_socket.recv(),
                    cv2.IMREAD_COLOR,
                )

                if decoded_rgb is not None:
                    latest_rgb = decoded_rgb

            if depth_socket in events:
                decoded_depth = decode(
                    depth_socket.recv(),
                    cv2.IMREAD_UNCHANGED,
                )

                if (
                    decoded_depth is not None
                    and decoded_depth.dtype == np.uint16
                ):
                    latest_depth = decoded_depth

            if raw_depth_socket in events:
                decoded_raw_depth = decode(
                    raw_depth_socket.recv(),
                    cv2.IMREAD_UNCHANGED,
                )

                if (
                    decoded_raw_depth is not None
                    and decoded_raw_depth.dtype == np.uint16
                ):
                    latest_raw_depth = decoded_raw_depth

            now = time.perf_counter()

            if (
                now >= next_item_time
                and latest_rgb is not None
                and latest_depth is not None
                and latest_raw_depth is not None
            ):
                writer.add_item(
                    colors={
                        "color_0": latest_rgb,
                    },
                    depths={
                        "depth_0": latest_depth,
                        "raw_depth_0": latest_raw_depth,
                    },
                    states=states,
                    actions=actions,
                )

                enqueued += 1

                backlog = (
                    writer.item_data_queue.qsize()
                )

                maximum_backlog = max(
                    maximum_backlog,
                    backlog,
                )

                next_item_time += period

                if next_item_time < now - period:
                    next_item_time = now + period

            if now >= next_report_time:
                elapsed = now - started

                print(
                    f"capture={enqueued / elapsed:5.1f} FPS  "
                    f"queued={enqueued:4d}  "
                    f"writer backlog="
                    f"{writer.item_data_queue.qsize():4d}"
                )

                next_report_time = now + 1.0

    except KeyboardInterrupt:
        print(
            "\nCapture interrupted; "
            "flushing queued data safely."
        )

    capture_stopped = time.perf_counter()

    capture_elapsed = max(
        capture_stopped - started,
        0.000001,
    )

    pending_at_stop = (
        writer.item_data_queue.qsize()
    )

    print("\nCapture stopped.")
    print(f"Items submitted:        {enqueued}")
    print(
        f"Submission rate:        "
        f"{enqueued / capture_elapsed:.2f} FPS"
    )
    print(f"Pending at stop:        {pending_at_stop}")
    print(
        f"Maximum writer backlog: {maximum_backlog}"
    )
    print(
        "Waiting for EpisodeWriter to drain and finalize..."
    )

    writer.save_episode()

    drain_started = time.perf_counter()
    next_drain_report = drain_started

    while not writer.is_ready():
        now = time.perf_counter()

        if now >= next_drain_report:
            print(
                f"Remaining queued items: "
                f"{writer.item_data_queue.qsize()}"
            )

            next_drain_report = now + 1.0

        time.sleep(0.01)

    drain_seconds = (
        time.perf_counter() - drain_started
    )

    writer.close()

    rgb_socket.close()
    depth_socket.close()
    raw_depth_socket.close()
    context.term()

    episode_dir = Path(
        writer.episode_dir
    )

    written_bytes = directory_size(
        episode_dir
    )

    color_count = len(
        list(
            (episode_dir / "colors").glob("*.jpg")
        )
    )

    depth_count = len(
        list(
            (episode_dir / "depths").glob("*.png")
        )
    )

    raw_depth_count = len(
        list(
            (episode_dir / "raw_depths").glob("*.png")
        )
    )

    print("\nFinal results")
    print("-------------")
    print(
        f"Capture duration:        "
        f"{capture_elapsed:.2f} s"
    )
    print(f"Items submitted:         {enqueued}")
    print(f"RGB files written:       {color_count}")
    print(f"Depth files written:     {depth_count}")
    print(
        f"Raw depth files written: {raw_depth_count}"
    )
    print(
        f"Pending at capture stop: {pending_at_stop}"
    )
    print(
        f"Maximum backlog:         {maximum_backlog}"
    )
    print(
        f"Post-stop drain time:    "
        f"{drain_seconds:.2f} s"
    )
    print(
        f"Episode size:            "
        f"{written_bytes / 1_000_000:.2f} MB"
    )
    print(
        f"Episode directory:       {episode_dir}"
    )

    all_files_written = (
        enqueued > 0
        and color_count == enqueued
        and depth_count == enqueued
        and raw_depth_count == enqueued
    )

    if (
        all_files_written
        and maximum_backlog <= 2
        and drain_seconds < 1.0
    ):
        print(
            "Result: EpisodeWriter kept up with all "
            "three streams at the requested rate."
        )
    elif enqueued > 0:
        print(
            "Result: EpisodeWriter experienced a backlog "
            "or did not write every submitted stream."
        )
    else:
        print(
            "Result: No complete RGB/aligned-depth/raw-depth "
            "items were submitted."
        )


if __name__ == "__main__":
    main()