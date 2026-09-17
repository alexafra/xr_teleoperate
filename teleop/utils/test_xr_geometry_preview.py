import threading

import numpy as np
import pytest

from teleop.utils.xr_geometry_preview import (
    GeometryPreviewContract,
    XrPreviewWorker,
    compose_xr_preview,
    normalize_xr_views,
    validate_geometry_preview_contract,
    validate_xr_preview_fps,
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
