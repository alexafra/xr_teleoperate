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
    CANONICAL_DEPTH_SCALE_M_PER_UNIT,
    DEPTH_FAR_M,
    DEPTH_NEAR_M,
    NORMAL_MAX_DEPTH_DELTA_M,
    color_intrinsics_for_preview,
    encode_depth_gray_rgb,
    encode_surface_normals_rgb,
)

XR_VIEW_CHOICES = (
    "rgb",
    "depth",
    "normals",
    "depth-overlay",
    "depth-edges",
    "depth-contours",
    "depth-wireframe",
    "near-warning",
    "normal-overlay",
    "normal-relight",
)
XR_FUSION_VIEWS = frozenset(
    {
        "depth-overlay",
        "depth-edges",
        "depth-contours",
        "depth-wireframe",
        "near-warning",
        "normal-overlay",
        "normal-relight",
    }
)
XR_COLOR_VIEWS = frozenset({"rgb", *XR_FUSION_VIEWS})
XR_GEOMETRY_VIEWS = frozenset({"depth", "normals", *XR_FUSION_VIEWS})
XR_NORMAL_VIEWS = frozenset({"normals", "normal-overlay", "normal-relight"})
DEFAULT_XR_PREVIEW_FPS = 30.0
DEFAULT_XR_FUSION_OPACITY = 0.5
DEFAULT_XR_NEAR_WARNING_M = 0.45
DEFAULT_XR_DEPTH_CONTOUR_SPACING_M = 0.05
XR_WIREFRAME_GRID_SPACING_PX = 32
XR_WIREFRAME_DEPTH_BANDS = 4


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


def xr_views_need_color(views: Sequence[str]) -> bool:
    """Whether a selection needs TeleImager's latest RGB slot."""

    return bool(set(views) & XR_COLOR_VIEWS)


def xr_views_need_geometry(views: Sequence[str]) -> bool:
    """Whether a selection needs TeleImager's latest aligned-depth slot."""

    return bool(set(views) & XR_GEOMETRY_VIEWS)


def xr_views_need_normals(views: Sequence[str]) -> bool:
    """Whether a selection needs calibrated v2 normal generation."""

    return bool(set(views) & XR_NORMAL_VIEWS)


def xr_views_use_fusion(views: Sequence[str]) -> bool:
    """Whether a selection combines RGB and geometry in one image."""

    return bool(set(views) & XR_FUSION_VIEWS)


def validate_xr_preview_fps(value: float) -> float:
    """Validate the bounded headset-preview refresh rate."""

    try:
        fps = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("--xr-preview-fps must be a number") from error
    if not np.isfinite(fps) or not 0.0 < fps <= 30.0:
        raise ValueError("--xr-preview-fps must be greater than 0 and at most 30")
    return fps


def validate_xr_fusion_opacity(value: float) -> float:
    """Validate the blend used by fused RGB/geometry views."""

    try:
        opacity = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("--xr-fusion-opacity must be a number") from error
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("--xr-fusion-opacity must be between 0 and 1")
    return opacity


def validate_xr_near_warning_m(value: float) -> float:
    """Validate the far boundary of the near-object warning band."""

    try:
        warning_m = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("--xr-near-warning-m must be a finite number") from error
    if not np.isfinite(warning_m) or not DEPTH_NEAR_M < warning_m <= DEPTH_FAR_M:
        raise ValueError(
            "--xr-near-warning-m must be greater than 0.25 and at most 1.0"
        )
    return warning_m


def validate_xr_depth_contour_spacing_m(value: float) -> float:
    """Validate equal optical-depth spacing inside the canonical range."""

    try:
        spacing_m = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "--xr-depth-contour-spacing-m must be a finite number"
        ) from error
    depth_span_m = DEPTH_FAR_M - DEPTH_NEAR_M
    if (
        not np.isfinite(spacing_m)
        or spacing_m < CANONICAL_DEPTH_SCALE_M_PER_UNIT
        or spacing_m > depth_span_m
    ):
        raise ValueError(
            "--xr-depth-contour-spacing-m must be at least 0.001 "
            "and at most 0.75"
        )
    return spacing_m


def validate_geometry_preview_contract(
    head_config: Mapping,
    views: Sequence[str],
) -> GeometryPreviewContract | None:
    """Fail closed on an unusable aligned-depth or calibration contract."""

    if not xr_views_need_geometry(views):
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
    if xr_views_need_normals(views):
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


def _validate_depth_image(image) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint16 or array.ndim != 2:
        raise ValueError(
            "Aligned depth must be an HxW uint16 image; "
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


def _validate_fusion_shapes(
    color_bgr: np.ndarray,
    depth: np.ndarray,
) -> None:
    if color_bgr.shape[:2] != depth.shape:
        raise ValueError(
            "Fused XR views require aligned RGB and depth with matching "
            f"dimensions; got {color_bgr.shape[:2]} and {depth.shape}"
        )


def _depth_valid_mask(
    depth: np.ndarray,
    *,
    depth_m: np.ndarray,
) -> np.ndarray:
    """Return the inclusive canonical metric-valid mask."""

    return (depth != 0) & (depth_m >= DEPTH_NEAR_M) & (depth_m <= DEPTH_FAR_M)


def _depth_meters(
    depth: np.ndarray,
    *,
    scale_m_per_unit: float,
) -> np.ndarray:
    """Apply the same canonicalized scale used by the geometry encoders."""

    try:
        scale = float(scale_m_per_unit)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Depth scale must be a finite number") from error
    if not np.isfinite(scale) or np.float32(scale) != np.float32(
        CANONICAL_DEPTH_SCALE_M_PER_UNIT
    ):
        raise ValueError(
            "Depth scale must match the canonical "
            f"{CANONICAL_DEPTH_SCALE_M_PER_UNIT} m/unit"
        )
    # Camera metadata can report the float32 expansion 0.001000000047... .
    # Canonical encoders intentionally snap that to exactly 0.001. Using the
    # same value in float64 also keeps integer-mm warning boundaries inclusive.
    return depth.astype(np.float64) * CANONICAL_DEPTH_SCALE_M_PER_UNIT


def _blend_masked(
    color_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    opacity: float,
) -> np.ndarray:
    """Alpha-blend only selected pixels, preserving all others byte-for-byte."""

    result = color_bgr.copy()
    if opacity == 0.0 or not np.any(mask):
        return result
    blended = cv2.addWeighted(
        color_bgr,
        1.0 - opacity,
        overlay_bgr,
        opacity,
        0.0,
    )
    result[mask] = blended[mask]
    return result


def _compose_depth_overlay(
    color_bgr: np.ndarray,
    depth_gray: np.ndarray,
    depth_valid: np.ndarray,
    *,
    opacity: float,
) -> np.ndarray:
    # Viridis is perceptually uniform and remains interpretable for common
    # forms of colour-vision deficiency. OpenCV returns the required BGR.
    depth_bgr = cv2.applyColorMap(depth_gray, cv2.COLORMAP_VIRIDIS)
    return _blend_masked(
        color_bgr,
        depth_bgr,
        depth_valid,
        opacity=opacity,
    )


def _compose_depth_edges(
    color_bgr: np.ndarray,
    depth_gray: np.ndarray,
    depth_valid: np.ndarray,
    *,
    opacity: float,
) -> np.ndarray:
    edges = cv2.Canny(depth_gray, 24, 64) != 0
    # Do not present missing/out-of-range depth boundaries as object geometry.
    # A contour is retained only when its complete 3x3 neighbourhood is valid.
    valid_interior = cv2.erode(
        depth_valid.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    edges &= valid_interior
    edge_bgr = np.empty_like(color_bgr)
    edge_bgr[:, :] = (255, 255, 0)  # bright cyan in the BGR buffer
    return _blend_masked(
        color_bgr,
        edge_bgr,
        edges,
        opacity=opacity,
    )


def _compose_depth_contours(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    *,
    spacing_m: float,
    opacity: float,
) -> np.ndarray:
    """Draw equal optical-depth isolines without painting object edges."""

    # Low-amplitude RealSense noise can alternate across an exact metric
    # threshold and otherwise paint most of a flat surface. A 5x5 Gaussian
    # stabilizes band assignment; the equally sized support mask below ensures
    # it cannot bridge a hole, out-of-range region, or object-depth jump.
    raw_depth_m = depth_m.astype(np.float32)
    contour_depth_m = raw_depth_m
    if min(depth_m.shape) >= 5:
        contour_depth_m = cv2.GaussianBlur(contour_depth_m, (5, 5), 0)
        kernel = np.ones((5, 5), dtype=np.uint8)
        full_valid_support = cv2.erode(
            depth_valid.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        local_min = cv2.erode(raw_depth_m, kernel, iterations=1)
        local_max = cv2.dilate(raw_depth_m, kernel, iterations=1)
        stable_support = full_valid_support & (
            (local_max - local_min) <= NORMAL_MAX_DEPTH_DELTA_M
        )
    else:
        stable_support = depth_valid
    # Quantize relative to the canonical near bound. A contour exists where
    # adjacent, continuous samples fall in different metric bands. The
    # epsilon only stabilizes exact millimetre boundaries at float precision.
    contour_band = np.floor(
        ((contour_depth_m - DEPTH_NEAR_M) / spacing_m) + 1e-6
    ).astype(np.int32)
    contours = np.zeros(depth_valid.shape, dtype=bool)

    horizontal_supported = (
        stable_support[:, :-1]
        & stable_support[:, 1:]
        & (
            np.abs(depth_m[:, :-1] - depth_m[:, 1:])
            <= NORMAL_MAX_DEPTH_DELTA_M
        )
    )
    horizontal_crossing = horizontal_supported & (
        contour_band[:, :-1] != contour_band[:, 1:]
    )
    contours[:, :-1] |= horizontal_crossing
    contours[:, 1:] |= horizontal_crossing

    vertical_supported = (
        stable_support[:-1, :]
        & stable_support[1:, :]
        & (
            np.abs(depth_m[:-1, :] - depth_m[1:, :])
            <= NORMAL_MAX_DEPTH_DELTA_M
        )
    )
    vertical_crossing = vertical_supported & (
        contour_band[:-1, :] != contour_band[1:, :]
    )
    contours[:-1, :] |= vertical_crossing
    contours[1:, :] |= vertical_crossing

    contour_bgr = np.empty_like(color_bgr)
    contour_bgr[:, :] = (255, 255, 0)  # bright cyan in the BGR buffer
    return _blend_masked(
        color_bgr,
        contour_bgr,
        contours,
        opacity=opacity,
    )


def _depth_continuity_mask(
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
) -> np.ndarray:
    """Keep pixels with a complete, locally continuous depth neighbourhood."""

    if min(depth_valid.shape) < 3:
        return np.zeros(depth_valid.shape, dtype=bool)
    continuous = np.zeros(depth_valid.shape, dtype=bool)
    center = depth_m[1:-1, 1:-1]
    supported = (
        depth_valid[1:-1, 1:-1]
        & depth_valid[1:-1, :-2]
        & depth_valid[1:-1, 2:]
        & depth_valid[:-2, 1:-1]
        & depth_valid[2:, 1:-1]
        & depth_valid[:-2, :-2]
        & depth_valid[:-2, 2:]
        & depth_valid[2:, :-2]
        & depth_valid[2:, 2:]
        & (np.abs(center - depth_m[1:-1, :-2]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[1:-1, 2:]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[:-2, 1:-1]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[2:, 1:-1]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[:-2, :-2]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[:-2, 2:]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[2:, :-2]) <= NORMAL_MAX_DEPTH_DELTA_M)
        & (np.abs(center - depth_m[2:, 2:]) <= NORMAL_MAX_DEPTH_DELTA_M)
    )
    continuous[1:-1, 1:-1] = supported
    return continuous


def _compose_depth_wireframe(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    *,
    opacity: float,
) -> np.ndarray:
    """Overlay a sparse depth-coloured screen-space triangular grid."""

    rows, columns = depth_valid.shape
    yy, xx = np.indices((rows, columns))
    spacing_px = XR_WIREFRAME_GRID_SPACING_PX
    grid = (
        (xx % spacing_px == 0)
        | (yy % spacing_px == 0)
        | ((xx + yy) % spacing_px == 0)
    )
    grid &= _depth_continuity_mask(depth_m, depth_valid)
    normalized_depth = np.clip(
        (depth_m - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M),
        0.0,
        1.0,
    )
    depth_band = np.minimum(
        (normalized_depth * XR_WIREFRAME_DEPTH_BANDS).astype(np.uint8),
        XR_WIREFRAME_DEPTH_BANDS - 1,
    )
    # Sample the perceptually ordered Viridis scale at four band centres.
    band_gray = (
        (depth_band.astype(np.float32) + 0.5)
        * (255.0 / XR_WIREFRAME_DEPTH_BANDS)
    ).astype(np.uint8)
    wireframe_bgr = cv2.applyColorMap(band_gray, cv2.COLORMAP_VIRIDIS)
    return _blend_masked(
        color_bgr,
        wireframe_bgr,
        grid,
        opacity=opacity,
    )


def _compose_near_warning(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    *,
    warning_m: float,
    opacity: float,
) -> np.ndarray:
    """Tint the valid warning band amber, progressing to red when nearer."""

    warning_mask = depth_valid & (depth_m <= warning_m)
    if opacity == 0.0 or not np.any(warning_mask):
        return color_bgr.copy()

    band_position = np.clip(
        (depth_m - DEPTH_NEAR_M) / (warning_m - DEPTH_NEAR_M),
        0.0,
        1.0,
    )
    warning_bgr = np.zeros_like(color_bgr)
    # RGB red at the near contract bound progresses to standard amber
    # (255, 191, 0) at the configurable warning boundary.
    warning_bgr[:, :, 1] = np.rint(191.0 * band_position).astype(np.uint8)
    warning_bgr[:, :, 2] = 255
    return _blend_masked(
        color_bgr,
        warning_bgr,
        warning_mask,
        opacity=opacity,
    )


def _compose_normal_overlay(
    color_bgr: np.ndarray,
    normals_rgb: np.ndarray,
    *,
    opacity: float,
) -> np.ndarray:
    """Blend canonical normal colours only where the normal is valid."""

    valid = np.any(normals_rgb != 0, axis=2)
    normals_bgr = np.ascontiguousarray(normals_rgb[:, :, ::-1])
    return _blend_masked(
        color_bgr,
        normals_bgr,
        valid,
        opacity=opacity,
    )


def _compose_normal_relight(
    color_bgr: np.ndarray,
    normals_rgb: np.ndarray,
    *,
    opacity: float,
) -> np.ndarray:
    valid = np.any(normals_rgb != 0, axis=2)
    if opacity == 0.0 or not np.any(valid):
        return color_bgr.copy()

    normals = (normals_rgb.astype(np.float32) - np.float32(1.0)) / np.float32(
        127.0
    ) - np.float32(1.0)
    # Canonical normals face the camera, so -Z is a neutral headlamp. Vary
    # only luminance: original RGB chroma and texture remain available.
    lambert = np.clip(-normals[:, :, 2], 0.0, 1.0)
    light_gain = np.float32(0.35) + np.float32(0.80) * lambert

    ycrcb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2YCrCb)
    source_y = ycrcb[:, :, 0].astype(np.float32)
    relit_y = np.clip(source_y * light_gain, 0.0, 255.0)
    mixed_y = (np.float32(1.0 - opacity) * source_y) + (np.float32(opacity) * relit_y)
    relit_ycrcb = ycrcb.copy()
    relit_ycrcb[:, :, 0][valid] = np.rint(mixed_y[valid]).astype(np.uint8)
    result = cv2.cvtColor(relit_ycrcb, cv2.COLOR_YCrCb2BGR)
    result[~valid] = color_bgr[~valid]
    return result


def compose_xr_preview(
    *,
    views: Sequence[str],
    output_shape: tuple[int, int],
    color_bgr=None,
    aligned_depth=None,
    geometry_contract: GeometryPreviewContract | None = None,
    fusion_opacity: float = DEFAULT_XR_FUSION_OPACITY,
    near_warning_m: float = DEFAULT_XR_NEAR_WARNING_M,
    depth_contour_spacing_m: float = DEFAULT_XR_DEPTH_CONTOUR_SPACING_M,
) -> np.ndarray:
    """Build one BGR frame for the existing TeleVuer image buffer."""

    normalized_views = normalize_xr_views(views)
    fusion_opacity = validate_xr_fusion_opacity(fusion_opacity)
    near_warning_m = validate_xr_near_warning_m(near_warning_m)
    depth_contour_spacing_m = validate_xr_depth_contour_spacing_m(
        depth_contour_spacing_m
    )
    output_height, output_width = output_shape
    if output_height <= 0 or output_width <= 0:
        raise ValueError("XR preview output dimensions must be positive")

    images: list[np.ndarray] = []
    if xr_views_need_color(normalized_views):
        if color_bgr is None:
            raise ValueError("RGB/fused XR preview has no color frame")
        rgb_bgr = _validate_bgr_image(color_bgr, name="RGB frame")
    else:
        rgb_bgr = None

    if xr_views_need_geometry(normalized_views):
        if aligned_depth is None:
            raise ValueError("Geometry XR preview has no aligned-depth frame")
        if geometry_contract is None:
            raise ValueError("Geometry XR preview has no validated contract")
        depth = _validate_depth_image(aligned_depth)
    else:
        depth = None

    if xr_views_use_fusion(normalized_views):
        _validate_fusion_shapes(rgb_bgr, depth)

    depth_rgb = None
    depth_m = None
    depth_valid = None
    if set(normalized_views) & {
        "depth",
        "depth-overlay",
        "depth-edges",
    }:
        depth_rgb = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=geometry_contract.depth_scale_m_per_unit,
        )
    if set(normalized_views) & {
        "depth-overlay",
        "depth-edges",
        "depth-contours",
        "depth-wireframe",
        "near-warning",
    }:
        depth_m = _depth_meters(
            depth,
            scale_m_per_unit=geometry_contract.depth_scale_m_per_unit,
        )
        depth_valid = _depth_valid_mask(
            depth,
            depth_m=depth_m,
        )

    normals_rgb = None
    if xr_views_need_normals(normalized_views):
        normals_rgb = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=geometry_contract.depth_scale_m_per_unit,
            color_intrinsics=geometry_contract.color_intrinsics,
        )

    for view in normalized_views:
        if view == "rgb":
            image_bgr = rgb_bgr
        elif view == "depth":
            # All channels are equal, so the canonical RGB depth encoding is
            # also the correct BGR buffer representation.
            image_bgr = depth_rgb
        elif view == "normals":
            # TeleVuer accepts BGR and performs its existing BGR->RGB copy.
            image_bgr = normals_rgb[:, :, ::-1]
        elif view == "depth-overlay":
            image_bgr = _compose_depth_overlay(
                rgb_bgr,
                depth_rgb[:, :, 0],
                depth_valid,
                opacity=fusion_opacity,
            )
        elif view == "depth-edges":
            image_bgr = _compose_depth_edges(
                rgb_bgr,
                depth_rgb[:, :, 0],
                depth_valid,
                opacity=fusion_opacity,
            )
        elif view == "depth-contours":
            image_bgr = _compose_depth_contours(
                rgb_bgr,
                depth_m,
                depth_valid,
                spacing_m=depth_contour_spacing_m,
                opacity=fusion_opacity,
            )
        elif view == "depth-wireframe":
            image_bgr = _compose_depth_wireframe(
                rgb_bgr,
                depth_m,
                depth_valid,
                opacity=fusion_opacity,
            )
        elif view == "near-warning":
            image_bgr = _compose_near_warning(
                rgb_bgr,
                depth_m,
                depth_valid,
                warning_m=near_warning_m,
                opacity=fusion_opacity,
            )
        elif view == "normal-overlay":
            image_bgr = _compose_normal_overlay(
                rgb_bgr,
                normals_rgb,
                opacity=fusion_opacity,
            )
        else:
            image_bgr = _compose_normal_relight(
                rgb_bgr,
                normals_rgb,
                opacity=fusion_opacity,
            )
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
        fusion_opacity: float = DEFAULT_XR_FUSION_OPACITY,
        near_warning_m: float = DEFAULT_XR_NEAR_WARNING_M,
        depth_contour_spacing_m: float = DEFAULT_XR_DEPTH_CONTOUR_SPACING_M,
        logger=None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._views = normalize_xr_views(views)
        self._output_shape = output_shape
        self._render = render
        self._read_color = read_color
        self._read_aligned_depth = read_aligned_depth
        self._geometry_contract = geometry_contract
        self._needs_color = xr_views_need_color(self._views)
        self._needs_geometry = xr_views_need_geometry(self._views)
        self._period_s = 1.0 / validate_xr_preview_fps(max_fps)
        self._fusion_opacity = validate_xr_fusion_opacity(fusion_opacity)
        self._near_warning_m = validate_xr_near_warning_m(near_warning_m)
        self._depth_contour_spacing_m = validate_xr_depth_contour_spacing_m(
            depth_contour_spacing_m
        )
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

        if self._needs_color and not callable(self._read_color):
            raise ValueError("RGB/fused XR preview requires a color-frame reader")
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
                color_bgr = self._read_color_bgr() if self._needs_color else None
                aligned_depth = (
                    self._read_aligned_depth() if self._needs_geometry else None
                )
                if (self._needs_color and color_bgr is None) or (
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
                    fusion_opacity=self._fusion_opacity,
                    near_warning_m=self._near_warning_m,
                    depth_contour_spacing_m=self._depth_contour_spacing_m,
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
