"""Hardware-free safety tests for the XR camera-only entry path."""

import ast
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import teleop_hand_and_arm

from teleop.utils.xr_geometry_preview import (
    GeometryPreviewContract,
    compose_xr_preview,
)


class _FakeImageClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def get_cam_config(self):
        return {
            "head_camera": {
                "binocular": False,
                "image_shape": [4, 12],
                "enable_zmq": True,
                "enable_webrtc": False,
                "webrtc_port": 60001,
            }
        }

    def get_head_frame(self):
        return None

    def get_head_depth_frame(self):
        return None

    def close(self):
        self.closed = True


class _FakeTeleVuer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def render_to_xr(self, image):
        del image

    def close(self):
        self.closed = True


class _FakePreviewWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


def _args():
    return SimpleNamespace(
        xr_view=("rgb", "near-warning"),
        xr_stereo=True,
        img_server_ip="192.0.2.10",
        display_mode="immersive",
        input_mode="hand",
        xr_preview_fps=30.0,
        xr_fusion_opacity=0.5,
        xr_near_warning_m=0.45,
        xr_depth_contour_spacing_m=0.05,
    )


class XrCameraOnlyTests(unittest.TestCase):
    def test_dichoptic_canvas_places_one_full_width_modality_per_eye(self):
        color = np.full((4, 12, 3), (10, 20, 30), dtype=np.uint8)
        depth = np.full((4, 12), 500, dtype=np.uint16)
        canvas = compose_xr_preview(
            views=("rgb", "depth"),
            output_shape=(4, 24),
            color_bgr=color,
            aligned_depth=depth,
            geometry_contract=GeometryPreviewContract(
                depth_scale_m_per_unit=0.001,
                color_intrinsics=None,
                calibration_source="test",
            ),
        )

        self.assertEqual(canvas.shape, (4, 24, 3))
        np.testing.assert_array_equal(canvas[:, :12], color)
        self.assertTrue(np.all(canvas[:, 12:, 0] == canvas[:, 12:, 1]))
        self.assertTrue(np.all(canvas[:, 12:, 1] == canvas[:, 12:, 2]))
        self.assertGreater(int(canvas[:, 12:, 0].min()), 0)

    def test_camera_only_stereo_path_never_touches_dds_or_robot_controllers(self):
        forbidden_calls = []

        def forbidden(name):
            def fail(*args, **kwargs):
                del args, kwargs
                forbidden_calls.append(name)
                raise AssertionError(f"camera-only mode called forbidden {name}")

            return fail

        forbidden_names = (
            "ChannelFactoryInitialize",
            "G1_29_ArmController",
            "G1_23_ArmController",
            "H1_2_ArmController",
            "H1_ArmController",
            "G1_29_ArmIK",
            "G1_23_ArmIK",
            "H1_2_ArmIK",
            "H1_ArmIK",
        )

        image_clients = []
        wrappers = []
        workers = []

        def image_client_factory(**kwargs):
            instance = _FakeImageClient(**kwargs)
            image_clients.append(instance)
            return instance

        def wrapper_factory(**kwargs):
            instance = _FakeTeleVuer(**kwargs)
            wrappers.append(instance)
            return instance

        def worker_factory(**kwargs):
            instance = _FakePreviewWorker(**kwargs)
            workers.append(instance)
            return instance

        def keyboard_listener(**kwargs):
            kwargs["on_press"]("q")

        with ExitStack() as patches:
            for name in forbidden_names:
                patches.enter_context(
                    mock.patch.object(
                        teleop_hand_and_arm,
                        name,
                        forbidden(name),
                    )
                )
            patches.enter_context(
                mock.patch.object(
                    teleop_hand_and_arm,
                    "validate_geometry_preview_contract",
                    return_value=SimpleNamespace(calibration_source="live"),
                )
            )
            self.assertEqual(
                teleop_hand_and_arm.run_xr_camera_only(
                    _args(),
                    image_client_factory=image_client_factory,
                    televuer_wrapper_factory=wrapper_factory,
                    preview_worker_factory=worker_factory,
                    keyboard_listener=keyboard_listener,
                    keyboard_stopper=lambda: None,
                ),
                0,
            )

        self.assertEqual(forbidden_calls, [])
        self.assertIs(wrappers[0].kwargs["binocular"], True)
        self.assertEqual(wrappers[0].kwargs["img_shape"], (4, 24))
        self.assertEqual(workers[0].kwargs["output_shape"], (4, 24))
        self.assertEqual(workers[0].kwargs["views"], ("rgb", "near-warning"))
        self.assertEqual(workers[0].kwargs["max_fps"], 30.0)
        self.assertEqual(workers[0].kwargs["fusion_opacity"], 0.5)
        self.assertEqual(workers[0].kwargs["near_warning_m"], 0.45)
        self.assertEqual(workers[0].kwargs["depth_contour_spacing_m"], 0.05)
        self.assertIs(image_clients[0].kwargs["eager_head_color"], True)
        self.assertIs(image_clients[0].kwargs["eager_aligned_depth"], True)
        self.assertIs(workers[0].started, True)
        self.assertIs(workers[0].closed, True)
        self.assertIs(image_clients[0].closed, True)
        self.assertIs(wrappers[0].closed, True)

    def test_main_dispatches_camera_only_before_dds_initialization(self):
        source_path = Path(teleop_hand_and_arm.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        dds_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ChannelFactoryInitialize"
        ]
        camera_only_exits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "SystemExit"
            and any(
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Name)
                and descendant.func.id == "run_xr_camera_only"
                for descendant in ast.walk(node.exc)
            )
        ]

        self.assertEqual(len(dds_calls), 2)
        self.assertEqual(len(camera_only_exits), 1)
        self.assertLess(
            camera_only_exits[0].lineno,
            min(call.lineno for call in dds_calls),
        )

    def test_near_warning_boundary_is_forwarded_in_both_preview_paths(self):
        source_path = Path(teleop_hand_and_arm.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        worker_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"preview_worker_factory", "XrPreviewWorker"}
        ]

        self.assertEqual(len(worker_calls), 2)
        for call in worker_calls:
            keyword = next(
                item for item in call.keywords if item.arg == "near_warning_m"
            )
            self.assertIsInstance(keyword.value, ast.Attribute)
            self.assertIsInstance(keyword.value.value, ast.Name)
            self.assertEqual(keyword.value.value.id, "args")
            self.assertEqual(keyword.value.attr, "xr_near_warning_m")

    def test_depth_contour_spacing_is_forwarded_in_both_preview_paths(self):
        source_path = Path(teleop_hand_and_arm.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        worker_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"preview_worker_factory", "XrPreviewWorker"}
        ]

        self.assertEqual(len(worker_calls), 2)
        for call in worker_calls:
            keyword = next(
                item
                for item in call.keywords
                if item.arg == "depth_contour_spacing_m"
            )
            self.assertIsInstance(keyword.value, ast.Attribute)
            self.assertIsInstance(keyword.value.value, ast.Name)
            self.assertEqual(keyword.value.value.id, "args")
            self.assertEqual(keyword.value.attr, "xr_depth_contour_spacing_m")


if __name__ == "__main__":
    unittest.main()
