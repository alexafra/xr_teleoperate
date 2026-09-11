from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import IntEnum
import operator
import threading
import time
from multiprocessing import Process, Array, Lock

from teleop.utils.subscriber_drop_tracker import SubscriberDropTracker

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

Inspire_Num_Motors = 6
INSPIRE_SUBSCRIBER_GAP_THRESHOLD_S = 0.075
INSPIRE_INITIAL_STATE_TIMEOUT_S = 5.0
UINT32_MAX = (1 << 32) - 1
INSPIRE_JOINT_ORDER = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)
INSPIRE_LEFT_JOINT_NAMES = (
    "kLeftHandPinky",
    "kLeftHandRing",
    "kLeftHandMiddle",
    "kLeftHandIndex",
    "kLeftHandThumbBend",
    "kLeftHandThumbRotation",
)
INSPIRE_RIGHT_JOINT_NAMES = (
    "kRightHandPinky",
    "kRightHandRing",
    "kRightHandMiddle",
    "kRightHandIndex",
    "kRightHandThumbBend",
    "kRightHandThumbRotation",
)
kTopicInspireDFXCommand = "rt/inspire/cmd"
kTopicInspireDFXState = "rt/inspire/state"


def _normalize_inspire_targets(targets):
    """Convert the six retargeted URDF angles to Inspire's open fraction."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (Inspire_Num_Motors,) or not np.all(np.isfinite(targets)):
        raise ValueError("Inspire retargeting must produce six finite joint values")

    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1])
    upper = np.array([1.7, 1.7, 1.7, 1.7, 0.5, 1.3])
    return np.clip((upper - targets) / (upper - lower), 0.0, 1.0)


def _validate_normalized_inspire_state(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (Inspire_Num_Motors,):
        raise ValueError(f"expected {Inspire_Num_Motors} motor positions")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains a non-finite motor position")
    if np.any(state < 0.0) or np.any(state > 1.0):
        raise ValueError("normalized motor position is outside [0, 1]")
    return state


def _validate_inspire_lost_counters(counters):
    if len(counters) != Inspire_Num_Motors:
        raise ValueError(
            f"expected {Inspire_Num_Motors} lost counters, got {len(counters)}"
        )

    validated = []
    for raw_counter in counters:
        if isinstance(raw_counter, bool):
            raise ValueError("lost counter must not be boolean")
        try:
            counter = operator.index(raw_counter)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("lost counter must be an integer") from error
        if not 0 <= counter <= UINT32_MAX:
            raise ValueError("lost counter is outside uint32 range")
        validated.append(counter)
    return tuple(validated)


class InspireDFXLostCounterTracker:
    """Classify the DFX bridge's six replicated per-hand loss counters."""

    _SIDES = ("left", "right")

    def __init__(self, clock_ns=None):
        self._clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
        self._lock = threading.Lock()
        self._baseline = {side: None for side in self._SIDES}
        self._last_observed = {side: None for side in self._SIDES}
        self._window_start_ns = None
        self._window_baseline_at_start = {side: None for side in self._SIDES}
        self._window_counts = self._new_counts()
        self._window_events = {side: [] for side in self._SIDES}
        self._window_malformed_messages = []

    def _new_counts(self):
        return {
            side: {
                "sample_count": 0,
                "baseline_count": 0,
                "accepted_count": 0,
                "held_sample_count": 0,
                "drop_event_count": 0,
                "lost_increment_count": 0,
                "reset_count": 0,
                "counter_regression_count": 0,
                "counter_divergence_count": 0,
            }
            for side in self._SIDES
        }

    @staticmethod
    def _transition(baseline, counters):
        # The official DFX bridge copies one failed side-read count into all six
        # MotorState.lost fields. Divergence is not a coherent side sample.
        if len(set(counters)) != 1:
            return "counter_divergence", False, None, 0
        if baseline is None:
            return "baseline", False, counters, 0
        if any(
            current < previous
            for current, previous in zip(counters, baseline, strict=True)
        ):
            # A service restart or uint32 wrap invalidates the old baseline.
            # The next coherent message establishes a new baseline; only the
            # following unchanged message is accepted.
            return "counter_regression", False, None, 0

        deltas = tuple(
            current - previous
            for current, previous in zip(counters, baseline, strict=True)
        )
        if any(deltas):
            # All six counters represent the same side-read failure count, so
            # use the common/max delta rather than multiplying the loss by six.
            return "lost_increment", False, counters, max(deltas)
        return "accepted", True, counters, 0

    def observe_combined(self, left_counters, right_counters, received_ns=None):
        counters_by_side = {
            "left": _validate_inspire_lost_counters(left_counters),
            "right": _validate_inspire_lost_counters(right_counters),
        }
        now_ns = self._clock_ns() if received_ns is None else int(received_ns)
        decisions = {}

        with self._lock:
            for side in self._SIDES:
                counters = counters_by_side[side]
                previous = self._baseline[side]
                event, accept, next_baseline, lost_increment = self._transition(
                    previous,
                    counters,
                )
                self._baseline[side] = next_baseline
                self._last_observed[side] = counters
                decisions[side] = {
                    "accept": accept,
                    "event": event,
                    "lost_increment": lost_increment,
                }

                if self._window_start_ns is None or now_ns < self._window_start_ns:
                    continue

                counts = self._window_counts[side]
                counts["sample_count"] += 1
                if accept:
                    counts["accepted_count"] += 1
                else:
                    counts["held_sample_count"] += 1
                if event == "baseline":
                    counts["baseline_count"] += 1
                elif event == "lost_increment":
                    counts["drop_event_count"] += 1
                    counts["lost_increment_count"] += lost_increment
                elif event == "counter_regression":
                    counts["reset_count"] += 1
                    counts["counter_regression_count"] += 1
                elif event == "counter_divergence":
                    counts["reset_count"] += 1
                    counts["counter_divergence_count"] += 1

                if event != "accepted":
                    self._window_events[side].append(
                        {
                            "offset_s": (now_ns - self._window_start_ns) / 1e9,
                            "event": event,
                            "previous_counters": (
                                None if previous is None else list(previous)
                            ),
                            "current_counters": list(counters),
                            "lost_increment": lost_increment,
                            "state_held": True,
                        }
                    )

        return decisions

    def observe_malformed(self, error, received_ns=None):
        now_ns = self._clock_ns() if received_ns is None else int(received_ns)
        with self._lock:
            if self._window_start_ns is None or now_ns < self._window_start_ns:
                return
            self._window_malformed_messages.append(
                {
                    "offset_s": (now_ns - self._window_start_ns) / 1e9,
                    "error": str(error),
                }
            )

    def begin_episode_window(self):
        with self._lock:
            if self._window_start_ns is not None:
                raise RuntimeError("Inspire DFX lost-counter window is already active")
            self._window_start_ns = self._clock_ns()
            self._window_baseline_at_start = dict(self._baseline)
            self._window_counts = self._new_counts()
            self._window_events = {side: [] for side in self._SIDES}
            self._window_malformed_messages = []

    def finish_episode_window(self):
        with self._lock:
            start_ns = self._window_start_ns
            if start_ns is None:
                raise RuntimeError("Inspire DFX lost-counter window is not active")
            end_ns = self._clock_ns()
            if end_ns < start_ns:
                raise RuntimeError("Inspire DFX lost-counter clock moved backwards")

            baseline_at_start = dict(self._window_baseline_at_start)
            counts = {
                side: dict(self._window_counts[side]) for side in self._SIDES
            }
            events = {
                side: list(self._window_events[side]) for side in self._SIDES
            }
            baseline_at_end = dict(self._baseline)
            last_observed = dict(self._last_observed)
            malformed_messages = list(self._window_malformed_messages)
            self._window_start_ns = None
            self._window_baseline_at_start = {
                side: None for side in self._SIDES
            }
            self._window_counts = self._new_counts()
            self._window_events = {side: [] for side in self._SIDES}
            self._window_malformed_messages = []

        summary = {
            "schema_version": 1,
            "metric": "inspire_dfx_motor_state_lost_counter",
            "definition": (
                "Per-side DFX read failures inferred from six identical "
                "MotorState.lost counters; q is held on increments, counter "
                "regressions, divergent counters, and baseline samples"
            ),
            "window_duration_s": (end_ns - start_ns) / 1e9,
            "malformed_message_count": len(malformed_messages),
            "malformed_messages": malformed_messages,
        }
        total_drop_events = 0
        total_lost_increments = 0
        total_resets = 0
        total_held_samples = 0
        for side in self._SIDES:
            side_summary = counts[side]
            side_summary.update(
                {
                    "baseline_at_start": (
                        None
                        if baseline_at_start[side] is None
                        else list(baseline_at_start[side])
                    ),
                    "baseline_at_end": (
                        None
                        if baseline_at_end[side] is None
                        else list(baseline_at_end[side])
                    ),
                    "last_observed_counters": (
                        None
                        if last_observed[side] is None
                        else list(last_observed[side])
                    ),
                    "events": events[side],
                }
            )
            summary[side] = side_summary
            total_drop_events += side_summary["drop_event_count"]
            total_lost_increments += side_summary["lost_increment_count"]
            total_resets += side_summary["reset_count"]
            total_held_samples += side_summary["held_sample_count"]

        summary["total_drop_event_count"] = total_drop_events
        summary["total_lost_increment_count"] = total_lost_increments
        summary["total_reset_count"] = total_resets
        summary["total_held_sample_count"] = total_held_samples
        summary["any_state_held"] = total_held_samples > 0
        summary["any_anomaly"] = (
            total_drop_events + total_resets + len(malformed_messages)
        ) > 0
        return summary


def _xr_motion_data_is_ready(
    xr_motion_data_ready_in,
    left_hand_data=None,
    right_hand_data=None,
):
    if xr_motion_data_ready_in is not None:
        with xr_motion_data_ready_in.get_lock():
            return bool(xr_motion_data_ready_in.value)

    # Standalone callers may not provide the shared readiness flag used by
    # teleop_hand_and_arm.py.  Fail closed until both XR skeletons themselves
    # pass the same validation, rather than treating a missing flag as ready.
    if left_hand_data is None or right_hand_data is None:
        return False
    return inspire_xr_hand_data_is_ready(left_hand_data, right_hand_data)


def inspire_xr_hand_data_is_ready(left_hand_data, right_hand_data):
    """Validate the legacy TeleVuer hand arrays when no ready flag is exposed."""
    left = np.asarray(left_hand_data)
    right = np.asarray(right_hand_data)
    expected_shape = (25, 3)

    def has_hand_geometry(hand):
        return np.max(np.linalg.norm(hand - hand[0], axis=1)) > 1e-4

    return bool(
        left.shape == expected_shape
        and right.shape == expected_shape
        and np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and not np.allclose(left, 0.0)
        and not np.allclose(right, 0.0)
        and has_hand_geometry(left)
        and has_hand_geometry(right)
    )


class Inspire_Controller_DFX:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
        xr_motion_data_ready_in=None,
        initial_state_timeout_s=INSPIRE_INITIAL_STATE_TIMEOUT_S,
        subscriber_gap_threshold_s=INSPIRE_SUBSCRIBER_GAP_THRESHOLD_S,
    ):
        logger_mp.info("Initialize Inspire_Controller_DFX...")
        if initial_state_timeout_s <= 0:
            raise ValueError("initial_state_timeout_s must be positive")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)


        # initialize handcmd publisher and handstate subscriber
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireDFXCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireDFXState, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.hand_state_lock = Lock()
        self.initial_state_received = threading.Event()
        self._initial_side_state_accepted = {"left": False, "right": False}
        self.subscriber_drop_tracker = SubscriberDropTracker(
            {"combined": kTopicInspireDFXState},
            subscriber_gap_threshold_s,
            stream_label="Inspire DFX state stream",
            window_label="Inspire DFX subscriber diagnostics",
        )
        self.lost_counter_tracker = InspireDFXLostCounterTracker()

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(
            target=self._subscribe_hand_state,
            name="inspire-dfx-state",
            daemon=True,
        )
        self.subscribe_state_thread.start()

        logger_mp.info("[Inspire_Controller_DFX] Waiting for a valid combined state...")
        if not self.initial_state_received.wait(float(initial_state_timeout_s)):
            raise TimeoutError(
                "Inspire DFX did not receive a valid combined hand state within "
                f"{float(initial_state_timeout_s):.2f}s"
            )
        logger_mp.info("[Inspire_Controller_DFX] Subscribe dds ok.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                self.hand_state_lock,
                xr_motion_data_ready_in,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_DFX OK!")

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                received_ns = time.monotonic_ns()
                try:
                    self._process_combined_state_message(hand_msg, received_ns)
                except (AttributeError, IndexError, TypeError, ValueError) as error:
                    try:
                        self.lost_counter_tracker.observe_malformed(
                            error,
                            received_ns,
                        )
                    except Exception as diagnostics_error:
                        logger_mp.error(
                            "[Inspire DFX state] Failed to record malformed-message "
                            f"diagnostics: {diagnostics_error}"
                        )
                    logger_mp.warning(
                        f"[Inspire DFX state] Ignoring invalid state message: {error}"
                    )
                    time.sleep(0.002)
                    continue
            time.sleep(0.002)

    @staticmethod
    def _decode_combined_sample(hand_msg):
        states = hand_msg.states
        expected_state_count = len(Inspire_Right_Hand_JointIndex) + len(
            Inspire_Left_Hand_JointIndex
        )
        if len(states) != expected_state_count:
            raise ValueError(
                f"expected {expected_state_count} combined motor states, got {len(states)}"
            )
        left_state = np.array(
            [states[joint_id].q for joint_id in Inspire_Left_Hand_JointIndex],
            dtype=np.float64,
        )
        right_state = np.array(
            [states[joint_id].q for joint_id in Inspire_Right_Hand_JointIndex],
            dtype=np.float64,
        )
        left_lost = _validate_inspire_lost_counters(
            [states[joint_id].lost for joint_id in Inspire_Left_Hand_JointIndex]
        )
        right_lost = _validate_inspire_lost_counters(
            [states[joint_id].lost for joint_id in Inspire_Right_Hand_JointIndex]
        )
        return (
            _validate_normalized_inspire_state(left_state),
            _validate_normalized_inspire_state(right_state),
            left_lost,
            right_lost,
        )

    @staticmethod
    def _decode_combined_state(hand_msg):
        left_state, right_state, _, _ = Inspire_Controller_DFX._decode_combined_sample(
            hand_msg
        )
        return left_state, right_state

    def _process_combined_state_message(self, hand_msg, received_ns=None):
        if received_ns is None:
            received_ns = time.monotonic_ns()
        left_state, right_state, left_lost, right_lost = self._decode_combined_sample(
            hand_msg
        )

        # A delivered, structurally valid combined sample advances DDS receive
        # diagnostics even when a service-side lost counter tells us to hold q.
        try:
            self.subscriber_drop_tracker.observe("combined", received_ns)
        except Exception as error:
            logger_mp.error(
                "[Inspire DFX state] Failed to record subscriber "
                f"diagnostics: {error}"
            )

        decisions = self.lost_counter_tracker.observe_combined(
            left_lost,
            right_lost,
            received_ns,
        )
        with self.hand_state_lock:
            if decisions["left"]["accept"]:
                with self.left_hand_state_array.get_lock():
                    self.left_hand_state_array[:] = left_state
                self._initial_side_state_accepted["left"] = True
            if decisions["right"]["accept"]:
                with self.right_hand_state_array.get_lock():
                    self.right_hand_state_array[:] = right_state
                self._initial_side_state_accepted["right"] = True

        if all(self._initial_side_state_accepted.values()):
            self.initial_state_received.set()
        return decisions

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        # Validate both sides before mutating or publishing the combined
        # message.  This keeps an invalid target from causing a partial update.
        left_q_target = _validate_normalized_inspire_state(left_q_target)
        right_q_target = _validate_normalized_inspire_state(right_q_target)
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):             
            self.hand_msg.cmds[id].q = left_q_target[idx]         
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):             
            self.hand_msg.cmds[id].q = right_q_target[idx] 

        self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        hand_state_lock=None,
        xr_motion_data_ready_in=None,
    ):
        self.running = True

        if hand_state_lock is None:
            left_q_target = np.array(left_hand_state_array[:], dtype=np.float64)
            right_q_target = np.array(right_hand_state_array[:], dtype=np.float64)
        else:
            with hand_state_lock:
                left_q_target = np.array(left_hand_state_array[:], dtype=np.float64)
                right_q_target = np.array(right_hand_state_array[:], dtype=np.float64)
        has_published_command = False

        # initialize inspire hand's cmd msg
        self.hand_msg  = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))]

        for idx, joint_id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = left_q_target[idx]
        for idx, joint_id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = right_q_target[idx]

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from one coherent DFX message.
                if hand_state_lock is None:
                    state_data = np.concatenate(
                        (np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:]))
                    )
                else:
                    with hand_state_lock:
                        state_data = np.concatenate(
                            (np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:]))
                        )

                if _xr_motion_data_is_ready(
                    xr_motion_data_ready_in,
                    left_hand_data,
                    right_hand_data,
                ):
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    left_q_target = _normalize_inspire_targets(left_q_target)
                    right_q_target = _normalize_inspire_targets(right_q_target)
                    has_published_command = True

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                # Do not send the historical all-open startup command. After the
                # first valid XR sample, continue holding the last target through
                # transient XR tracking loss.
                if has_published_command:
                    self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_DFX has been closed.")



kTopicInspireFTPLeftCommand   = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand  = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState  = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"


def get_inspire_end_effector_info(protocol):
    """Return the canonical raw-episode contract for an Inspire hand pair."""
    if protocol not in ("dfx", "ftp"):
        raise ValueError(f"Unsupported Inspire protocol: {protocol}")
    return {
        "schema_version": 1,
        "type": "inspire",
        "protocol": protocol,
        "hand_dof": Inspire_Num_Motors,
        "value_unit": "normalized_open_fraction",
        "value_range": [0.0, 1.0],
        "zero_semantics": "fully_closed",
        "one_semantics": "fully_open",
        "left_joint_names": list(INSPIRE_LEFT_JOINT_NAMES),
        "right_joint_names": list(INSPIRE_RIGHT_JOINT_NAMES),
        "canonical_order": "left_then_right",
    }


class Inspire_Controller_FTP:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
        xr_motion_data_ready_in=None,
        initial_state_timeout_s=INSPIRE_INITIAL_STATE_TIMEOUT_S,
        subscriber_gap_threshold_s=INSPIRE_SUBSCRIBER_GAP_THRESHOLD_S,
    ):
        logger_mp.info("Initialize Inspire_Controller_FTP...")
        if initial_state_timeout_s <= 0:
            raise ValueError("initial_state_timeout_s must be positive")
        try:
            from inspire_sdkpy import inspire_dds
            import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default
        except ImportError as error:
            raise RuntimeError(
                "inspire_ftp requires the vendor inspire_sdkpy package"
            ) from error
        self._new_ftp_control_message = inspire_hand_default.get_inspire_hand_ctrl
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)


        # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.initial_state_received = {
            "left": threading.Event(),
            "right": threading.Event(),
        }
        self.subscriber_drop_tracker = SubscriberDropTracker(
            {
                "left": kTopicInspireFTPLeftState,
                "right": kTopicInspireFTPRightState,
            },
            subscriber_gap_threshold_s,
            stream_label="Inspire FTP hand side",
            window_label="Inspire FTP subscriber diagnostics",
            aggregate_count_key="total_side_gap_count",
        )

        # A blocking read on one hand must not prevent the healthy hand from
        # refreshing its state or diagnostics.
        self.left_subscribe_state_thread = threading.Thread(
            target=self._subscribe_hand_state,
            args=(
                self.LeftHandState_subscriber,
                self.left_hand_state_array,
                "left",
            ),
            name="inspire-ftp-left-state",
            daemon=True,
        )
        self.right_subscribe_state_thread = threading.Thread(
            target=self._subscribe_hand_state,
            args=(
                self.RightHandState_subscriber,
                self.right_hand_state_array,
                "right",
            ),
            name="inspire-ftp-right-state",
            daemon=True,
        )
        self.left_subscribe_state_thread.start()
        self.right_subscribe_state_thread.start()

        logger_mp.info("[Inspire_Controller_FTP] Waiting for valid left and right states...")
        deadline = time.monotonic() + float(initial_state_timeout_s)
        for side in ("left", "right"):
            remaining_s = max(0.0, deadline - time.monotonic())
            if not self.initial_state_received[side].wait(remaining_s):
                missing = [
                    name
                    for name, event in self.initial_state_received.items()
                    if not event.is_set()
                ]
                raise TimeoutError(
                    "Inspire FTP did not receive valid state for "
                    f"{', '.join(missing)} within {float(initial_state_timeout_s):.2f}s"
                )
        logger_mp.info("[Inspire_Controller_FTP] Initial hand states received.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                xr_motion_data_ready_in,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self, subscriber, state_array, side):
        logger_mp.info(f"[Inspire_Controller_FTP] {side} subscribe thread started.")
        while True:
            state_msg = subscriber.Read()
            if state_msg is not None:
                received_ns = time.monotonic_ns()
                try:
                    state = self._decode_ftp_state(state_msg)
                except (AttributeError, TypeError, ValueError) as error:
                    logger_mp.warning(
                        f"[Inspire FTP {side} state] Ignoring invalid message: {error}"
                    )
                    time.sleep(0.002)
                    continue

                try:
                    self.subscriber_drop_tracker.observe(side, received_ns)
                except Exception as error:
                    logger_mp.error(
                        f"[Inspire FTP {side} state] Failed to record subscriber "
                        f"diagnostics: {error}"
                    )

                with state_array.get_lock():
                    state_array[:] = state
                self.initial_state_received[side].set()
            time.sleep(0.002)

    @staticmethod
    def _decode_ftp_state(state_msg):
        angle_act = state_msg.angle_act
        if len(angle_act) != Inspire_Num_Motors:
            raise ValueError(
                f"expected {Inspire_Num_Motors} angle_act values, got {len(angle_act)}"
            )
        state = np.asarray(angle_act, dtype=np.float64) / 1000.0
        return _validate_normalized_inspire_state(state)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        def validate_scaled_command(values, side):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (Inspire_Num_Motors,):
                raise ValueError(
                    f"expected {Inspire_Num_Motors} {side} command values"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{side} command contains a non-finite value")
            if np.any(values < 0.0) or np.any(values > 1000.0):
                raise ValueError(f"{side} command is outside [0, 1000]")
            if not np.all(values == np.floor(values)):
                raise ValueError(f"{side} command must contain integer values")
            return [int(value) for value in values]

        # Validate both messages before writing either side.
        left_angle_cmd_scaled = validate_scaled_command(
            left_angle_cmd_scaled,
            "left",
        )
        right_angle_cmd_scaled = validate_scaled_command(
            right_angle_cmd_scaled,
            "right",
        )

        # Left Hand Command
        left_cmd_msg = self._new_ftp_control_message()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # Right Hand Command
        right_cmd_msg = self._new_ftp_control_message()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)

        # 临时打开前 N 次的 log
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 50:
            logger_mp.info(f"[Inspire_Controller_FTP] Publish cmd L={left_angle_cmd_scaled} R={right_angle_cmd_scaled} ")
            self._debug_count += 1


    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        xr_motion_data_ready_in=None,
    ):
        logger_mp.info("[Inspire_Controller_FTP] Control process started.")
        self.running = True

        left_q_target = np.array(left_hand_state_array[:], dtype=np.float64)
        right_q_target = np.array(right_hand_state_array[:], dtype=np.float64)
        has_published_command = False

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if _xr_motion_data_is_ready(
                    xr_motion_data_ready_in,
                    left_hand_data,
                    right_hand_data,
                ):
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    left_q_target = _normalize_inspire_targets(left_q_target)
                    right_q_target = _normalize_inspire_targets(right_q_target)
                    has_published_command = True

                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target]

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                if has_published_command:
                    self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_FTP has been closed.")

# Update hand state, according to the official documentation:
# 1. https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# 2. https://support.unitree.com/home/en/G1_developer/inspire_ftp_dexterity_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
