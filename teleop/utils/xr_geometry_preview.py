"""Non-blocking XR previews for RGB and aligned-depth geometry views.

The worker deliberately polls TeleImager's latest-only subscriber slots.  It
does not own a frame queue, so slow surface-normal generation drops
intermediate camera frames instead of creating latency in robot control.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
from teleimager.geometry_preview import (
    color_intrinsics_for_preview,
    encode_depth_gray_rgb,
    encode_surface_normals_rgb,
)


XR_VIEW_CHOICES = ("rgb", "depth", "normals")
DEFAULT_XR_PREVIEW_FPS = 15.0


@dataclass(frozen=True)
class GeometryPreviewContract:
    """Validated inputs needed by the canonical geometry encoders."""

    depth_scale_m_per_unit: float
    color_intrinsics: dict | None
    calibration_source: str | None


def normalize_xr_views(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Return an ordered, duplicate-free XR view selection.

    Omitting the option preserves the historical RGB headset view.
    """

    if not requested:
        return ("rgb",)

    views = tuple(requested)
    unknown = [view for view in views if view not in XR_VIEW_CHOICES]
    if unknown:
        raise ValueError(f"Unknown XR view: {unknown[0]!r}")
    if len(set(views)) != len(views):
        raise ValueError("Each --xr-view modality may be selected only once")
    return views


def validate_xr_preview_fps(value: float) -> float:
    """Validate the bounded headset-preview refresh rate."""

    try:
        fps = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("--xr-preview-fps must be a number") from error
    if not np.isfinite(fps) or not 0.0 < fps <= 30.0:
        raise ValueError("--xr-preview-fps must be greater than 0 and at most 30")
    return fps


def validate_geometry_preview_contract(
    head_config: Mapping,
    views: Sequence[str],
) -> GeometryPreviewContract | None:
    """Fail closed on an unusable aligned-depth or calibration contract."""

    geometry_views = set(views) & {"depth", "normals"}
    if not geometry_views:
        return None
    if not isinstance(head_config, Mapping):
        raise ValueError("Head-camera config must be a mapping")
    if not head_config.get("enable_depth", False):
        raise ValueError("The selected XR geometry view requires aligned depth")
    depth_port = head_config.get("depth_zmq_port")
    if (
        isinstance(depth_port, bool)
        or not isinstance(depth_port, (int, np.integer))
        or not 1 <= int(depth_port) <= 65535
    ):
        raise ValueError(
            "The selected XR geometry view requires a valid aligned-depth ZMQ port"
        )

    scale = head_config.get("depth_scale_m_per_unit")
    # Exercise the public canonical encoder at startup rather than duplicating
    # its float32-exact scale contract here.
    encode_depth_gray_rgb(
        np.zeros((1, 1), dtype=np.uint16),
        scale_m_per_unit=scale,
    )

    color_intrinsics = None
    calibration_source = None
    if "normals" in geometry_views:
        color_intrinsics, calibration_source = color_intrinsics_for_preview(head_config)
        image_shape = head_config.get("image_shape")
        expected_shape = (
            color_intrinsics["height"],
            color_intrinsics["width"],
        )
        if (
            not isinstance(image_shape, (list, tuple))
            or tuple(image_shape) != expected_shape
        ):
            raise ValueError(
                "Aligned-depth image_shape does not match calibrated color "
                f"intrinsics: {image_shape!r} vs {expected_shape!r}"
            )

    return GeometryPreviewContract(
        depth_scale_m_per_unit=float(scale),
        color_intrinsics=color_intrinsics,
        calibration_source=calibration_source,
    )


def _validate_bgr_image(image, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name} must be an HxWx3 uint8 image; "
            f"got shape={array.shape}, dtype={array.dtype}"
        )
    return array


def _fit_in_panel(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Aspect-fit one BGR image in a fixed black panel."""

    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def compose_xr_preview(
    *,
    views: Sequence[str],
    output_shape: tuple[int, int],
    color_bgr=None,
    aligned_depth=None,
    geometry_contract: GeometryPreviewContract | None = None,
) -> np.ndarray:
    """Build one BGR frame for the existing TeleVuer image buffer."""

    normalized_views = normalize_xr_views(views)
    output_height, output_width = output_shape
    if output_height <= 0 or output_width <= 0:
        raise ValueError("XR preview output dimensions must be positive")

    images: list[np.ndarray] = []
    if "rgb" in normalized_views:
        if color_bgr is None:
            raise ValueError("RGB XR preview has no color frame")
        rgb_bgr = _validate_bgr_image(color_bgr, name="RGB frame")
    else:
        rgb_bgr = None

    if set(normalized_views) & {"depth", "normals"}:
        if aligned_depth is None:
            raise ValueError("Geometry XR preview has no aligned-depth frame")
        if geometry_contract is None:
            raise ValueError("Geometry XR preview has no validated contract")
        depth = np.asarray(aligned_depth)
    else:
        depth = None

    for view in normalized_views:
        if view == "rgb":
            image_bgr = rgb_bgr
        elif view == "depth":
            # All channels are equal, so the canonical RGB depth encoding is
            # also the correct BGR buffer representation.
            image_bgr = encode_depth_gray_rgb(
                depth,
                scale_m_per_unit=geometry_contract.depth_scale_m_per_unit,
            )
        else:
            normals_rgb = encode_surface_normals_rgb(
                depth,
                scale_m_per_unit=geometry_contract.depth_scale_m_per_unit,
                color_intrinsics=geometry_contract.color_intrinsics,
            )
            # TeleVuer accepts BGR and performs its existing BGR->RGB copy.
            image_bgr = normals_rgb[:, :, ::-1]
        images.append(np.ascontiguousarray(image_bgr))

    if len(images) == 1:
        image = images[0]
        if image.shape[:2] != output_shape:
            image = _fit_in_panel(
                image,
                height=output_height,
                width=output_width,
            )
        return np.ascontiguousarray(image)

    # Preserve the configured TeleVuer buffer shape. Each selected modality is
    # aspect-fitted into an equal-width panel in CLI order.
    boundaries = np.linspace(0, output_width, len(images) + 1, dtype=int)
    panels = [
        _fit_in_panel(
            image,
            height=output_height,
            width=int(boundaries[index + 1] - boundaries[index]),
        )
        for index, image in enumerate(images)
    ]
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


class XrPreviewWorker:
    """Fetch, encode, and render only the latest camera frames off-loop."""

    def __init__(
        self,
        *,
        views: Sequence[str],
        output_shape: tuple[int, int],
        render: Callable[[np.ndarray], None],
        read_color: Callable[[], object] | None,
        read_aligned_depth: Callable[[], object] | None,
        geometry_contract: GeometryPreviewContract | None,
        max_fps: float = DEFAULT_XR_PREVIEW_FPS,
        logger=None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._views = normalize_xr_views(views)
        self._output_shape = output_shape
        self._render = render
        self._read_color = read_color
        self._read_aligned_depth = read_aligned_depth
        self._geometry_contract = geometry_contract
        self._needs_geometry = bool(set(self._views) & {"depth", "normals"})
        self._period_s = 1.0 / validate_xr_preview_fps(max_fps)
        self._logger = logger or logging.getLogger(__name__)
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="xr-preview",
            daemon=True,
        )
        self._started = False
        self._last_error: str | None = None

        if "rgb" in self._views and not callable(self._read_color):
            raise ValueError("RGB XR preview requires a color-frame reader")
        if self._needs_geometry:
            if not callable(self._read_aligned_depth):
                raise ValueError("Geometry XR preview requires an aligned-depth reader")
            if self._geometry_contract is None:
                raise ValueError(
                    "Geometry XR preview requires a validated geometry contract"
                )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("XR preview worker has already been started")
        self._started = True
        self._thread.start()

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._logger.warning(
                    "XR preview worker did not stop within %.1f seconds",
                    timeout,
                )

    def _read_color_bgr(self):
        frame = self._read_color()
        return None if frame is None else getattr(frame, "bgr", frame)

    def _run(self) -> None:
        while not self._stop.is_set():
            cycle_started = self._monotonic()
            try:
                color_bgr = self._read_color_bgr() if "rgb" in self._views else None
                aligned_depth = (
                    self._read_aligned_depth() if self._needs_geometry else None
                )
                if ("rgb" in self._views and color_bgr is None) or (
                    self._needs_geometry and aligned_depth is None
                ):
                    self._wait_until_next_cycle(cycle_started)
                    continue

                preview = compose_xr_preview(
                    views=self._views,
                    output_shape=self._output_shape,
                    color_bgr=color_bgr,
                    aligned_depth=aligned_depth,
                    geometry_contract=self._geometry_contract,
                )
                self._render(preview)
                if self._last_error is not None:
                    self._logger.info(
                        "XR preview recovered after: %s", self._last_error
                    )
                    self._last_error = None
            except Exception as error:
                # Preview is ancillary. Hold the last TeleVuer frame and keep
                # control/recording alive, but never substitute another view.
                detail = f"{type(error).__name__}: {error}"
                if detail != self._last_error:
                    self._logger.warning(
                        "XR preview frame dropped; holding the last rendered "
                        "frame (%s)",
                        detail,
                    )
                    self._last_error = detail

            self._wait_until_next_cycle(cycle_started)

    def _wait_until_next_cycle(self, cycle_started: float) -> None:
        remaining = self._period_s - (self._monotonic() - cycle_started)
        if remaining > 0.0:
            self._stop.wait(remaining)
