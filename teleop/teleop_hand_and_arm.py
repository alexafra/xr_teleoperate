import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleimager.geometry_preview import legacy_depth_metadata_from_head_config
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.episode_voice_feedback import (
    AsyncEpisodeVoiceNotifier,
    EpisodeRecordingController,
)
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.rgbd_capture import (
    EXPERIMENTAL_ATOMIC_RGBD_RECORDING,
    PreferAtomicHeadRgbdCapture,
    should_prefer_atomic_rgbd_recording,
)
from teleop.utils.xr_geometry_preview import (
    DEFAULT_XR_PREVIEW_FPS,
    XR_VIEW_CHOICES,
    XrPreviewWorker,
    normalize_xr_views,
    validate_geometry_preview_contract,
    validate_xr_preview_fps,
)
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_CONTROLLER = None
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: an edge-triggered S press requests a transition.
#  --> auto  : saving remains interlocked until EpisodeWriter is ready.

def on_press(key):
    global STOP, START
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START and RECORD_CONTROLLER is not None:
        RECORD_CONTROLLER.request_key_press()
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def on_release(key):
    if key == 's' and RECORD_CONTROLLER is not None:
        RECORD_CONTROLLER.request_key_release()


def on_ipc_press(key):
    """IPC commands are discrete taps rather than held keyboard keys."""
    if key == 's' and START and RECORD_CONTROLLER is not None:
        RECORD_CONTROLLER.request_tap()
    else:
        on_press(key)

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument(
        '--xr-view',
        action='append',
        choices=XR_VIEW_CHOICES,
        default=None,
        help=(
            'Headset image modality. Repeat to create a left-to-right split '
            'view; omitted means the existing RGB view.'
        ),
    )
    parser.add_argument(
        '--xr-preview-fps',
        type=float,
        default=DEFAULT_XR_PREVIEW_FPS,
        help='Maximum local headset-preview rate (0 < FPS <= 30; default 15)',
    )
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument(
        '--episode-voice-feedback',
        action='store_true',
        help='Speak successful episode start, stop, and completed-save events',
    )
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    if args.ee in ("inspire_dfx", "inspire_ftp") and args.input_mode != "hand":
        parser.error(f"--ee {args.ee} requires --input-mode hand")
    if args.episode_voice_feedback and not args.record:
        parser.error("--episode-voice-feedback requires --record")
    try:
        args.xr_view = normalize_xr_views(args.xr_view)
        args.xr_preview_fps = validate_xr_preview_fps(args.xr_preview_fps)
    except ValueError as error:
        parser.error(str(error))
    logger_mp.info(f"args: {args}")
    tactile_reader = None
    voice_notifier = None
    recording_controller = None
    recorder = None
    xr_preview_worker = None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_ipc_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={
                    "on_press": on_press,
                    "on_release": on_release,
                    "until": None,
                    "sequential": True,
                },
                daemon=True,
            )
            listen_keyboard_thread.start()

        # image client
        geometry_view_requested = bool(
            set(args.xr_view) & {"depth", "normals"}
        )
        img_client = ImageClient(
            host=args.img_server_ip,
            request_bgr=True,
            eager_head_color=args.record or "rgb" in args.xr_view,
            eager_aligned_depth=args.record or geometry_view_requested,
            eager_raw_depth=args.record,
        )
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        head_config = camera_config["head_camera"]
        if geometry_view_requested and args.display_mode == "pass-through":
            raise RuntimeError(
                "--xr-view depth/normals cannot be displayed in pass-through mode"
            )
        if geometry_view_requested and head_config.get("binocular", False):
            raise RuntimeError(
                "Aligned-depth XR geometry preview currently requires a "
                "monocular head-camera configuration"
            )
        if geometry_view_requested and not head_config.get("enable_zmq", False):
            raise RuntimeError(
                "--xr-view depth/normals requires the head-camera ZMQ transport"
            )
        try:
            geometry_preview_contract = validate_geometry_preview_contract(
                head_config,
                args.xr_view,
            )
        except ValueError as error:
            raise RuntimeError(
                f"Invalid XR geometry-preview contract: {error}"
            ) from error
        if (
            geometry_preview_contract is not None
            and geometry_preview_contract.calibration_source == "pinned"
        ):
            logger_mp.warning(
                "XR normals use the exact pinned 640x480@30 color intrinsics "
                "for D435I serial 254322071415 because the camera server "
                "omitted calibration metadata."
            )

        # WebRTC carries the camera's RGB stream directly. Explicit geometry
        # must use TeleVuer's local shared-image buffer instead.
        headset_webrtc = bool(
            head_config.get("enable_webrtc", False)
            and not geometry_view_requested
        )
        if geometry_view_requested and head_config.get("enable_webrtc", False):
            logger_mp.info(
                "XR geometry selected: using the local ZMQ/TeleVuer buffer "
                "instead of the camera RGB WebRTC stream."
            )
        rgbd_capture = None
        if should_prefer_atomic_rgbd_recording(
            recording_enabled=args.record,
            end_effector=args.ee,
            head_config=head_config,
            experimental_opt_in=EXPERIMENTAL_ATOMIC_RGBD_RECORDING,
        ):
            try:
                from teleimager.image_client import decode_rgbd_frame
            except Exception:
                # Older TeleImager clients remain supported through the
                # helper's legacy fallback path.
                decode_rgbd_frame = None
            rgbd_capture = PreferAtomicHeadRgbdCapture(
                img_client,
                head_config,
                decode_atomic_frame=decode_rgbd_frame,
                logger=logger_mp,
            )
        xr_need_local_img = not (
            args.display_mode == 'pass-through' or headset_webrtc
        )

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=headset_webrtc,
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )

        if xr_need_local_img:
            xr_preview_worker = XrPreviewWorker(
                views=args.xr_view,
                output_shape=tuple(head_config['image_shape']),
                render=tv_wrapper.render_to_xr,
                read_color=(
                    img_client.get_head_frame
                    if "rgb" in args.xr_view
                    else None
                ),
                read_aligned_depth=(
                    img_client.get_head_depth_frame
                    if geometry_view_requested
                    else None
                ),
                geometry_contract=geometry_preview_contract,
                max_fps=args.xr_preview_fps,
                logger=logger_mp,
            )
            xr_preview_worker.start()
            logger_mp.info(
                "XR preview active: %s at up to %.1f FPS",
                "+".join(args.xr_view),
                args.xr_preview_fps,
            )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        hand_ctrl = None
        inspire_end_effector_info = None
        inspire_xr_motion_data_ready = None
        if args.ee in ("inspire_dfx", "inspire_ftp"):
            inspire_xr_motion_data_ready = Value('b', False, lock=True)
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import (
                Inspire_Controller_DFX,
                get_inspire_end_effector_info,
                inspire_xr_hand_data_is_ready,
            )
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                simulation_mode=args.sim,
                xr_motion_data_ready_in=inspire_xr_motion_data_ready,
            )
            inspire_end_effector_info = get_inspire_end_effector_info("dfx")
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import (
                Inspire_Controller_FTP,
                get_inspire_end_effector_info,
                inspire_xr_hand_data_is_ready,
            )
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                simulation_mode=args.sim,
                xr_motion_data_ready_in=inspire_xr_motion_data_ready,
            )
            inspire_end_effector_info = get_inspire_end_effector_info("ftp")
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            if args.ee == "inspire_ftp" and not args.sim:
                from teleop.robot_control.inspire_tactile import (
                    InspireTactileReader,
                    TACTILE_PADS,
                )

                tactile_reader = InspireTactileReader()
                if not tactile_reader.wait_for_data(timeout=3.0):
                    raise RuntimeError(
                        "Inspire FTP recording requires valid tactile data from both hands"
                    )

            episode_diagnostics_sources = {}
            subscriber_drop_tracker = getattr(
                hand_ctrl,
                "subscriber_drop_tracker",
                None,
            )
            if subscriber_drop_tracker is not None:
                episode_diagnostics_sources[
                    f"{args.ee}_state_subscribers"
                ] = subscriber_drop_tracker
            lost_counter_tracker = getattr(
                hand_ctrl,
                "lost_counter_tracker",
                None,
            )
            if lost_counter_tracker is not None:
                episode_diagnostics_sources[
                    f"{args.ee}_lost_counters"
                ] = lost_counter_tracker

            depth_scale = None
            depth_scale_reported = None
            depth_calibration = None
            if head_config.get("enable_depth", False):
                required_depth_metadata = (
                    "depth_scale_m_per_unit",
                    "depth_scale_reported_m_per_unit",
                    "calibration",
                )
                missing_depth_metadata = [
                    key
                    for key in required_depth_metadata
                    if head_config.get(key) is None
                ]
                if missing_depth_metadata:
                    try:
                        legacy_metadata = legacy_depth_metadata_from_head_config(
                            head_config
                        )
                    except ValueError as error:
                        raise RuntimeError(
                            "Depth recording requires calibration metadata from "
                            "the updated live TeleImager server; missing "
                            + ", ".join(missing_depth_metadata)
                        ) from error
                    logger_mp.warning(
                        "Camera server omitted calibration metadata; recording "
                        "with the exact pinned calibration for D435I serial "
                        "254322071415. This startup-only metadata fallback does "
                        "not change frame capture or control-loop processing."
                    )
                    depth_scale = legacy_metadata["depth_scale_m_per_unit"]
                    depth_scale_reported = legacy_metadata[
                        "depth_scale_reported_m_per_unit"
                    ]
                    depth_calibration = legacy_metadata["calibration"]
                else:
                    depth_scale = head_config["depth_scale_m_per_unit"]
                    depth_scale_reported = head_config[
                        "depth_scale_reported_m_per_unit"
                    ]
                    depth_calibration = head_config["calibration"]
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless,
                                     depth_scale_m_per_unit=depth_scale,
                                     depth_scale_reported_m_per_unit=depth_scale_reported,
                                     depth_calibration=depth_calibration,
                                     episode_diagnostics_sources=episode_diagnostics_sources,
                                     end_effector_info=inspire_end_effector_info,)
            if tactile_reader is not None:
                recorder.info["tactile_names"]["left_ee"] = list(TACTILE_PADS)
                recorder.info["tactile_names"]["right_ee"] = list(TACTILE_PADS)
            if rgbd_capture is not None:
                recorder.info["rgbd_pairing"] = rgbd_capture.episode_metadata()
            if args.episode_voice_feedback:
                voice_notifier = AsyncEpisodeVoiceNotifier()
            recording_controller = EpisodeRecordingController(
                recorder,
                notifier=voice_notifier,
            )
            RECORD_CONTROLLER = recording_controller

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if rgbd_capture is not None:
                # Preserve the experimental recorder transport's startup
                # probing. Headset preview acquisition runs independently.
                rgbd_capture.read()

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        # main loop. robot start to follow VR user's motion
        head_depth = None
        raw_head_depth = None

        while not STOP:
            start_time = time.time()
            head_img = None
            record_head_bgr = None
            rgbd_pairing = None
            # get image
            if rgbd_capture is not None:
                rgbd_sample = rgbd_capture.read()
                record_head_bgr = rgbd_sample.color_bgr
                head_depth = rgbd_sample.aligned_depth
                rgbd_pairing = rgbd_sample.pairing
            elif camera_config['head_camera']['enable_zmq']:
                if args.record:
                    head_img = img_client.get_head_frame()
                    if head_img is not None:
                        record_head_bgr = head_img.bgr
            if (args.record and camera_config["head_camera"].get("enable_depth", False)):
                if rgbd_capture is None:
                    head_depth = img_client.get_head_depth_frame()

                if camera_config["head_camera"].get("raw_depth_zmq_port") is not None:
                    head_raw_depth = img_client.get_head_raw_depth_frame()
                else:
                    head_raw_depth = None
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # Record transitions are edge-triggered and remain interlocked
            # while EpisodeWriter finishes an asynchronous save.
            if args.record:
                record_event = recording_controller.update()
                RECORD_RUNNING = recording_controller.recording
                READY = recording_controller.ready_to_start
                if record_event == "start_failed":
                    logger_mp.error("Failed to create episode. Recording not started.")
                elif record_event == "stopping":
                    logger_mp.info(
                        f"Capture stopped. Pending frames: "
                        f"{recorder.item_data_queue.qsize()}"
                    )
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            if inspire_xr_motion_data_ready is not None:
                xr_ready = getattr(tele_data, "motion_data_ready", None)
                if xr_ready is None:
                    xr_ready = inspire_xr_hand_data_is_ready(
                        tele_data.left_hand_pos,
                        tele_data.right_hand_pos,
                    )
                with inspire_xr_motion_data_ready.get_lock():
                    inspire_xr_motion_data_ready.value = bool(xr_ready)
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                READY = recording_controller.ready_to_start
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config["head_camera"].get("enable_depth", False):
                        if head_depth is not None:
                            depths["depth_0"] = head_depth
                        else:
                            logger_mp.warning("Head aligned-depth image is None!")

                        if camera_config["head_camera"].get("raw_depth_zmq_port") is not None:
                            if head_raw_depth is not None:
                                depths["raw_depth_0"] = head_raw_depth
                            else:
                                logger_mp.warning("Head raw-depth image is None!")
                    if camera_config['head_camera']['binocular']:
                        if record_head_bgr is not None:
                            colors[f"color_{0}"] = record_head_bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = record_head_bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if record_head_bgr is not None:
                            colors[f"color_{0}"] = record_head_bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    tactiles = (
                        tactile_reader.get_tactiles()
                        if tactile_reader is not None
                        else None
                    )
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            tactiles=tactiles,
                            sim_state=sim_state,
                            rgbd_pairing=rgbd_pairing,
                        )
                    else:
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            tactiles=tactiles,
                            rgbd_pairing=rgbd_pairing,
                        )

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if args.record and recording_controller is not None:
                # Freeze episode timing and subscriber diagnostics before any
                # shutdown homing or queue-drain delay. The controller avoids
                # duplicating a save that is already pending.
                recording_controller.stop_for_shutdown()
                RECORD_RUNNING = recording_controller.recording
            elif args.record and recorder is not None and not recorder.is_ready():
                # Retain the legacy fallback if initialization failed between
                # constructing EpisodeWriter and its recording controller.
                recorder.save_episode()
        except Exception as e:
            logger_mp.error(f"Failed to stop active recording: {e}")

        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if xr_preview_worker is not None:
                xr_preview_worker.close()
        except Exception as e:
            logger_mp.error(f"Failed to close XR preview worker: {e}")

        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record and recorder is not None:
                recorder.close()
                if recording_controller is not None:
                    recording_controller.poll_save_completion()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        try:
            RECORD_CONTROLLER = None
            if voice_notifier is not None:
                voice_notifier.close()
        except Exception as e:
            logger_mp.error(f"Failed to close episode voice feedback: {e}")

        try:
            if tactile_reader is not None:
                tactile_reader.close()
        except Exception as e:
            logger_mp.error(f"Failed to close Inspire tactile reader: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
