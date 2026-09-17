import threading

import cv2
import numpy as np
import pytest

import teleop.utils.xr_geometry_preview as preview_module
from teleop.utils.xr_geometry_preview import (
    DEFAULT_XR_DEPTH_CONTOUR_SPACING_M,
    DEFAULT_XR_FUSION_OPACITY,
    DEFAULT_XR_NEAR_WARNING_M,
    DEFAULT_XR_PREVIEW_FPS,
    GeometryPreviewContract,
    XrPreviewWorker,
    compose_xr_preview,
    normalize_xr_views,
    validate_geometry_preview_contract,
    validate_xr_depth_contour_spacing_m,
    validate_xr_fusion_opacity,
    validate_xr_near_warning_m,
    validate_xr_preview_fps,
    xr_views_need_color,
    xr_views_need_geometry,
    xr_views_need_normals,
    xr_views_use_fusion,
)

SMALL_INTRINSICS = {
    "width": 6,
    "height": 4,
    "fx": 6.0,
    "fy": 6.0,
    "cx": 2.5,
    "cy": 1.5,
}
SMALL_CONTRACT = GeometryPreviewContract(
    depth_scale_m_per_unit=0.001,
    color_intrinsics=SMALL_INTRINSICS,
    calibration_source="test",
)


def test_default_preview_rate_is_30_hz():
    assert DEFAULT_XR_PREVIEW_FPS == 30.0


def test_geometry_preview_defaults():
    assert DEFAULT_XR_NEAR_WARNING_M == 0.45
    assert DEFAULT_XR_FUSION_OPACITY == 0.5


def test_default_view_preserves_rgb_bytes():
    color_bgr = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    preview = compose_xr_preview(
        views=normalize_xr_views(None),
        output_shape=(4, 6),
        color_bgr=color_bgr,
    )

    np.testing.assert_array_equal(preview, color_bgr)


def test_normals_are_converted_to_bgr_for_existing_televuer_buffer():
    depth = np.full((4, 6), 600, dtype=np.uint16)

    preview = compose_xr_preview(
        views=("normals",),
        output_shape=(4, 6),
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
    )

    # Canonical flat-plane normal RGB is [128, 128, 1]. TeleVuer's input is
    # BGR, so the buffer receives the reversed triplet.
    assert preview[1, 2].tolist() == [1, 128, 128]
    assert preview[0, 0].tolist() == [0, 0, 0]


def test_split_view_keeps_output_shape_and_cli_order():
    color_bgr = np.full((4, 6, 3), [7, 11, 19], dtype=np.uint8)
    depth = np.full((4, 6), 600, dtype=np.uint16)

    preview = compose_xr_preview(
        views=("rgb", "depth"),
        output_shape=(4, 6),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
    )

    assert preview.shape == (4, 6, 3)
    assert preview.dtype == np.uint8
    # Each 4:3 source is aspect-fitted into a 3x4 panel with black bars.
    assert preview[1, 1].tolist() == [7, 11, 19]
    assert preview[1, 4, 0] > 0
    assert preview[0].sum() == 0


def test_fusion_view_stream_requirements_are_centralized():
    for view in (
        "depth-overlay",
        "depth-edges",
        "depth-contours",
        "depth-wireframe",
        "near-warning",
        "normal-overlay",
        "normal-relight",
    ):
        assert xr_views_use_fusion((view,))
        assert xr_views_need_color((view,))
        assert xr_views_need_geometry((view,))
    assert not xr_views_need_normals(
        (
            "depth-overlay",
            "depth-edges",
            "depth-contours",
            "depth-wireframe",
            "near-warning",
        )
    )
    assert xr_views_need_normals(("normal-overlay", "normal-relight"))


def test_depth_overlay_changes_only_metric_valid_pixels():
    color_bgr = np.full((4, 6, 3), [13, 37, 71], dtype=np.uint8)
    depth = np.tile(
        np.array([0, 249, 250, 600, 1000, 1001], dtype=np.uint16),
        (4, 1),
    )

    preview = compose_xr_preview(
        views=("depth-overlay",),
        output_shape=(4, 6),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(preview[:, [0, 1, 5]], color_bgr[:, [0, 1, 5]])
    assert np.all(np.any(preview[:, [2, 3, 4]] != color_bgr[:, [2, 3, 4]], axis=2))


def test_depth_edges_keep_invalid_boundary_unpainted_but_show_valid_step():
    color_bgr = np.full((8, 12, 3), [20, 40, 60], dtype=np.uint8)
    invalid_boundary = np.full((8, 12), 500, dtype=np.uint16)
    invalid_boundary[:, 6:] = 0

    invalid_preview = compose_xr_preview(
        views=("depth-edges",),
        output_shape=(8, 12),
        color_bgr=color_bgr,
        aligned_depth=invalid_boundary,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(invalid_preview, color_bgr)

    valid_step = np.full((8, 12), 400, dtype=np.uint16)
    valid_step[:, 6:] = 800
    step_preview = compose_xr_preview(
        views=("depth-edges",),
        output_shape=(8, 12),
        color_bgr=color_bgr,
        aligned_depth=valid_step,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    changed = np.any(step_preview != color_bgr, axis=2)
    assert np.any(changed[:, 5:7])
    assert not np.any(changed[:, :4])
    assert not np.any(changed[:, 8:])


def test_depth_contours_show_metric_crossing_but_not_depth_jump():
    color_bgr = np.full((9, 14, 3), [20, 40, 60], dtype=np.uint8)
    depth = np.tile(
        np.array(
            [349, 349, 349, 349, 349, 351, 351, 351, 351, 351, 600, 600, 600, 0],
            dtype=np.uint16,
        ),
        (9, 1),
    )

    preview = compose_xr_preview(
        views=("depth-contours",),
        output_shape=(9, 14),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    changed = np.any(preview != color_bgr, axis=2)
    assert np.all(changed[2:-2, 4:6])
    assert not np.any(changed[:, :3])
    assert not np.any(changed[:, 7:])


def test_depth_contours_filter_isolated_threshold_speckle():
    color_bgr = np.full((7, 7, 3), [20, 40, 60], dtype=np.uint8)
    depth = np.full((7, 7), 349, dtype=np.uint16)
    depth[3, 3] = 351

    preview = compose_xr_preview(
        views=("depth-contours",),
        output_shape=(7, 7),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(preview, color_bgr)


def test_depth_contours_do_not_paint_threshold_checkerboard_noise():
    color_bgr = np.full((65, 65, 3), [20, 40, 60], dtype=np.uint8)
    yy, xx = np.indices((65, 65))
    depth = np.where((xx + yy) % 2 == 0, 499, 501).astype(np.uint16)

    preview = compose_xr_preview(
        views=("depth-contours",),
        output_shape=(65, 65),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(preview, color_bgr)


def test_depth_wireframe_breaks_at_invalid_pixels_and_depth_jumps():
    color_bgr = np.full((66, 66, 3), [20, 40, 60], dtype=np.uint8)
    depth = np.full((66, 66), 400, dtype=np.uint16)
    depth[:, 32:] = 800
    depth[16, 16] = 0

    preview = compose_xr_preview(
        views=("depth-wireframe",),
        output_shape=(66, 66),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(preview[16, 16], color_bgr[16, 16])
    np.testing.assert_array_equal(preview[10, 32], color_bgr[10, 32])
    np.testing.assert_array_equal(preview[10, 50], color_bgr[10, 50])
    assert np.any(preview[10, 64] != color_bgr[10, 64])


def test_depth_wireframe_breaks_diagonal_segments_at_diagonal_jump():
    color_bgr = np.full((66, 66, 3), [20, 40, 60], dtype=np.uint8)
    depth = np.full((66, 66), 440, dtype=np.uint16)
    depth[10, 22] = 400
    depth[11, 21] = 480

    preview = compose_xr_preview(
        views=("depth-wireframe",),
        output_shape=(66, 66),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(preview[10, 22], color_bgr[10, 22])
    np.testing.assert_array_equal(preview[11, 21], color_bgr[11, 21])


def test_depth_wireframe_uses_four_ordered_depth_colour_bands():
    color_bgr = np.full((66, 66, 3), [20, 40, 60], dtype=np.uint8)
    depth = np.tile(
        np.linspace(250, 1000, 66, dtype=np.uint16),
        (66, 1),
    )

    preview = compose_xr_preview(
        views=("depth-wireframe",),
        output_shape=(66, 66),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    line_colours = np.unique(preview[32, 1:-1], axis=0)
    assert line_colours.shape[0] == 4


def test_near_warning_progresses_red_to_amber_and_preserves_other_pixels():
    color_bgr = np.full((2, 8, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.tile(
        np.array([0, 249, 250, 350, 450, 451, 1000, 1001], dtype=np.uint16),
        (2, 1),
    )

    preview = compose_xr_preview(
        views=("near-warning",),
        output_shape=(2, 8),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(
        preview[:, [0, 1, 5, 6, 7]],
        color_bgr[:, [0, 1, 5, 6, 7]],
    )
    np.testing.assert_array_equal(preview[:, 2], [[0, 0, 255]] * 2)
    np.testing.assert_array_equal(preview[:, 3], [[0, 96, 255]] * 2)
    np.testing.assert_array_equal(preview[:, 4], [[0, 191, 255]] * 2)


def test_near_warning_boundary_is_configurable():
    color_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    preview = compose_xr_preview(
        views=("near-warning",),
        output_shape=(1, 1),
        color_bgr=color_bgr,
        aligned_depth=np.array([[450]], dtype=np.uint16),
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
        near_warning_m=0.65,
    )

    assert preview[0, 0].tolist() == [0, 96, 255]


def test_near_warning_uses_default_fusion_opacity_for_tint_intensity():
    color_bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    preview = compose_xr_preview(
        views=("near-warning",),
        output_shape=(1, 1),
        color_bgr=color_bgr,
        aligned_depth=np.array([[250]], dtype=np.uint16),
        geometry_contract=SMALL_CONTRACT,
    )
    expected = cv2.addWeighted(
        color_bgr,
        1.0 - DEFAULT_XR_FUSION_OPACITY,
        np.array([[[0, 0, 255]]], dtype=np.uint8),
        DEFAULT_XR_FUSION_OPACITY,
        0.0,
    )

    np.testing.assert_array_equal(preview, expected)


def test_normal_overlay_blends_canonical_colors_only_on_valid_normals():
    color_bgr = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.full((4, 6), 600, dtype=np.uint16)
    preview = compose_xr_preview(
        views=("normal-overlay",),
        output_shape=(4, 6),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=0.5,
    )
    expected_valid = cv2.addWeighted(
        color_bgr[1:2, 2:3],
        0.5,
        np.array([[[1, 128, 128]]], dtype=np.uint8),
        0.5,
        0.0,
    )[0, 0]

    np.testing.assert_array_equal(preview[0], color_bgr[0])
    np.testing.assert_array_equal(preview[-1], color_bgr[-1])
    np.testing.assert_array_equal(preview[1, 2], expected_valid)


def test_normal_relight_changes_only_luminance_on_v2_valid_mask():
    color_bgr = np.full((4, 6, 3), [30, 80, 150], dtype=np.uint8)
    depth = np.full((4, 6), 600, dtype=np.uint16)

    preview = compose_xr_preview(
        views=("normal-relight",),
        output_shape=(4, 6),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=1.0,
    )

    np.testing.assert_array_equal(preview[0], color_bgr[0])
    np.testing.assert_array_equal(preview[-1], color_bgr[-1])
    delta = preview[1, 2].astype(int) - color_bgr[1, 2].astype(int)
    assert np.all(delta > 0)
    assert np.ptp(delta) <= 2
    before_ycrcb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2YCrCb)
    after_ycrcb = cv2.cvtColor(preview, cv2.COLOR_BGR2YCrCb)
    np.testing.assert_allclose(
        after_ycrcb[1, 2, 1:],
        before_ycrcb[1, 2, 1:],
        atol=1,
    )


@pytest.mark.parametrize(
    "view",
    [
        "depth-overlay",
        "depth-edges",
        "depth-contours",
        "depth-wireframe",
        "near-warning",
        "normal-overlay",
        "normal-relight",
    ],
)
def test_zero_fusion_opacity_preserves_rgb_exactly(view):
    color_bgr = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.full((4, 6), 600, dtype=np.uint16)

    preview = compose_xr_preview(
        views=(view,),
        output_shape=(4, 6),
        color_bgr=color_bgr,
        aligned_depth=depth,
        geometry_contract=SMALL_CONTRACT,
        fusion_opacity=0.0,
    )

    np.testing.assert_array_equal(preview, color_bgr)


def test_fused_view_rejects_misaligned_runtime_shapes():
    with pytest.raises(ValueError, match="matching dimensions"):
        compose_xr_preview(
            views=("depth-overlay",),
            output_shape=(4, 6),
            color_bgr=np.zeros((4, 6, 3), dtype=np.uint8),
            aligned_depth=np.zeros((3, 6), dtype=np.uint16),
            geometry_contract=SMALL_CONTRACT,
        )


def test_multiple_views_encode_each_geometry_representation_once(monkeypatch):
    calls = {"depth": 0, "normals": 0}
    original_depth = preview_module.encode_depth_gray_rgb
    original_normals = preview_module.encode_surface_normals_rgb

    def count_depth(*args, **kwargs):
        calls["depth"] += 1
        return original_depth(*args, **kwargs)

    def count_normals(*args, **kwargs):
        calls["normals"] += 1
        return original_normals(*args, **kwargs)

    monkeypatch.setattr(preview_module, "encode_depth_gray_rgb", count_depth)
    monkeypatch.setattr(
        preview_module,
        "encode_surface_normals_rgb",
        count_normals,
    )
    compose_xr_preview(
        views=(
            "depth",
            "depth-overlay",
            "depth-edges",
            "depth-contours",
            "depth-wireframe",
            "near-warning",
            "normals",
            "normal-overlay",
            "normal-relight",
        ),
        output_shape=(4, 54),
        color_bgr=np.full((4, 6, 3), 80, dtype=np.uint8),
        aligned_depth=np.full((4, 6), 600, dtype=np.uint16),
        geometry_contract=SMALL_CONTRACT,
    )

    assert calls == {"depth": 1, "normals": 1}


def test_geometry_contract_requires_aligned_depth_transport():
    with pytest.raises(ValueError, match="aligned-depth ZMQ port"):
        validate_geometry_preview_contract(
            {
                "enable_depth": True,
                "depth_scale_m_per_unit": 0.001,
                "depth_zmq_port": None,
            },
            ("depth",),
        )


def test_normals_contract_accepts_exact_pinned_inspire_camera():
    contract = validate_geometry_preview_contract(
        {
            "type": "realsense",
            "serial_number": "254322071415",
            "image_shape": [480, 640],
            "fps": 30,
            "enable_depth": True,
            "depth_zmq_port": 5558,
            "depth_scale_m_per_unit": 0.0010000000474974513,
        },
        ("normals",),
    )

    assert contract.calibration_source == "pinned"
    assert contract.color_intrinsics["width"] == 640
    assert contract.depth_scale_m_per_unit == 0.0010000000474974513


@pytest.mark.parametrize("fps", [0, -1, 30.01, float("inf"), float("nan")])
def test_preview_rate_is_bounded(fps):
    with pytest.raises(ValueError, match="at most 30"):
        validate_xr_preview_fps(fps)


@pytest.mark.parametrize(
    "opacity",
    [-0.01, 1.01, float("inf"), float("nan")],
)
def test_fusion_opacity_is_finite_and_bounded(opacity):
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_xr_fusion_opacity(opacity)


@pytest.mark.parametrize(
    "warning_m",
    [0.25, 0.0, -1.0, 1.0001, float("inf"), float("nan")],
)
def test_near_warning_boundary_is_finite_and_bounded(warning_m):
    with pytest.raises(ValueError, match="greater than 0.25 and at most 1.0"):
        validate_xr_near_warning_m(warning_m)


@pytest.mark.parametrize("warning_m", [0.2500001, 0.45, 1.0])
def test_near_warning_boundary_accepts_open_closed_range(warning_m):
    assert validate_xr_near_warning_m(warning_m) == warning_m


def test_depth_contour_spacing_default_and_bounds():
    assert DEFAULT_XR_DEPTH_CONTOUR_SPACING_M == 0.05
    assert validate_xr_depth_contour_spacing_m(0.001) == 0.001
    assert validate_xr_depth_contour_spacing_m(0.75) == 0.75
    for spacing_m in (0.0009, 0.7501, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="at least 0.001 and at most 0.75"):
            validate_xr_depth_contour_spacing_m(spacing_m)


def test_worker_reads_only_selected_stream_and_renders_off_thread():
    rendered = []
    rendered_event = threading.Event()

    def read_color():
        raise AssertionError("depth-only preview must not read RGB")

    def read_depth():
        return np.full((4, 6), 600, dtype=np.uint16)

    def render(image):
        rendered.append((threading.current_thread().name, image.copy()))
        rendered_event.set()

    worker = XrPreviewWorker(
        views=("depth",),
        output_shape=(4, 6),
        render=render,
        read_color=read_color,
        read_aligned_depth=read_depth,
        geometry_contract=SMALL_CONTRACT,
        max_fps=30,
    )
    worker.start()
    try:
        assert rendered_event.wait(timeout=1.0)
    finally:
        worker.close()

    assert rendered[0][0] == "xr-preview"
    assert rendered[0][1].shape == (4, 6, 3)


def test_duplicate_view_is_rejected_instead_of_rendered_twice():
    with pytest.raises(ValueError, match="only once"):
        normalize_xr_views(("rgb", "rgb"))
