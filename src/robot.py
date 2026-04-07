"""
Robot control thread for the UFactory xArm6.

Connects to the arm over IP and runs a velocity control loop that
maps the 3D position offset vector (from the vision thread) to robot
motion in two modes:

    Translation mode (xArm Mode 5 — Cartesian velocity):
        The end-effector tracks the user's hand position in XYZ.

    Orientation mode (xArm Mode 4 — joint velocity):
        The wrist joints (J4/J5 or J6) track hand rotation.

Safety measures applied at startup:
    - Reduced max TCP speed (100 mm/s)
    - Reduced Cartesian workspace bounding box
    - Collision sensitivity enabled
    - Self-collision detection enabled
    - TCP tool model set to a cylinder matching the gripper geometry
"""

import time
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../xArm-Python-SDK'))
from xarm.wrapper import XArmAPI

from state import SystemState

# --- Robot configuration ---
ROBOT_IP = "192.168.1.213"

# Gripper digital I/O channels
GRIPPER_CLOSE_CH = 1
GRIPPER_OPEN_CH = 0
GRIPPER_PULSE_S = 0.8   # Duration of the pneumatic pulse

# Workspace bounding box [x_min, y_min, z_min, x_max, y_max, z_max] mm
# (xArm reduced mode format: [x+, y-, z, x-, y+, z+])
WORKSPACE = [600, -115, 350, -350, 1000, 0]

# TCP tool geometry (cylinder approximating the gripper)
TOOL_RADIUS_MM = 115
TOOL_HEIGHT_MM = 200
TCP_OFFSET = [0, 0, 130, 0, 0, 0]   # mm, degrees


def _scale_velocity(error: float) -> float:
    """
    Map a position error (mm) to a velocity command (mm/s).

    Dead zone below 5 mm, then proportional ramp with increasing
    gain for larger errors. Keeps motion smooth at close range and
    responsive when far from target.

    Args:
        error: signed distance to target in one axis (mm).

    Returns:
        Signed velocity command (mm/s).
    """
    abs_err = abs(error)
    sign = 1 if error >= 0 else -1

    if abs_err < 5:
        return 0.0
    elif abs_err < 20:
        return sign * 0.10 * abs_err
    elif abs_err < 40:
        return sign * 0.30 * abs_err
    elif abs_err < 70:
        return sign * 0.60 * abs_err
    elif abs_err < 100:
        return sign * 1.25 * abs_err
    else:
        return sign * 1.50 * abs_err


def _connect_and_home(arm: XArmAPI) -> None:
    """
    Enable the arm, apply safety settings, and move to home position.

    Raises:
        RuntimeError: if the arm reports an error after connection.
    """
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)

    arm.move_gohome(wait=True)
    arm.reset(wait=True)

    # Safety
    arm.set_collision_sensitivity(True)
    arm.set_self_collision_detection(True)
    arm.set_collision_tool_model(21, TOOL_RADIUS_MM, TOOL_HEIGHT_MM)
    arm.set_tcp_offset(TCP_OFFSET)

    # Velocity control mode
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.1)

    # Reduced speed + workspace
    arm.set_reduced_max_tcp_speed(100)
    arm.set_tcp_maxacc(300)
    arm.set_reduced_tcp_boundary(WORKSPACE)
    arm.set_reduced_mode(True)


def _switch_to_translation(arm: XArmAPI) -> tuple[float, float, float]:
    """
    Switch arm to Mode 5 (Cartesian velocity) and capture current position.

    Returns:
        (x, y, z) reference position in mm.
    """
    arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0])
    time.sleep(0.1)
    arm.set_mode(0); arm.set_state(0); time.sleep(0.1)
    arm.set_mode(5); arm.set_state(0); time.sleep(0.1)
    x, y, z, *_ = arm.position
    return x, y, z


def _switch_to_orientation(arm: XArmAPI) -> tuple[float, float, float,
                                                    float, float, float, int]:
    """
    Switch arm to Mode 4 (joint velocity) and capture current joint angles.

    Returns:
        (j1, j2, j3, j4, j5, j6, code) joint angles in degrees.
    """
    arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
    time.sleep(0.1)
    arm.set_mode(0); arm.set_state(0); time.sleep(0.1)
    arm.set_mode(4); arm.set_state(0); time.sleep(0.1)
    arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0])
    time.sleep(0.1)
    return arm.angles


def _toggle_gripper(arm: XArmAPI, state: SystemState) -> None:
    """
    Fire a pneumatic pulse to open or close the gripper,
    then update shared state.
    """
    time.sleep(1)
    if state.gripper_open:
        arm.set_tgpio_digital(GRIPPER_CLOSE_CH, 1)
        time.sleep(GRIPPER_PULSE_S)
        arm.set_tgpio_digital(GRIPPER_CLOSE_CH, 0)
        arm.set_tgpio_digital(GRIPPER_OPEN_CH, 0)
        with state.lock:
            state.gripper_open = False
    else:
        arm.set_tgpio_digital(GRIPPER_OPEN_CH, 1)
        time.sleep(GRIPPER_PULSE_S)
        arm.set_tgpio_digital(GRIPPER_CLOSE_CH, 0)
        arm.set_tgpio_digital(GRIPPER_OPEN_CH, 0)
        with state.lock:
            state.gripper_open = True


def _safe_shutdown(arm: XArmAPI, state: SystemState) -> None:
    """
    Stop all motion, open gripper if closed, and return to home.
    """
    with state.lock:
        gripper_open = state.gripper_open
        translation = state.translation_mode

    if not gripper_open:
        _toggle_gripper(arm, state)

    if translation:
        arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
    else:
        arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0])

    arm.reset(wait=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)
    arm.move_gohome(wait=True)
    arm.disconnect()
    print("[robot] Safe shutdown complete.")


def run_robot_thread(state: SystemState) -> None:
    """
    Robot control loop (runs in its own thread).

    Connects to the xArm6, then continuously reads position offsets
    from shared state and sends velocity commands to track the user's
    hand in real time.

    Args:
        state: shared SystemState instance.
    """
    arm = XArmAPI(ROBOT_IP)

    try:
        _connect_and_home(arm)
    except Exception as e:
        print(f"[robot] Connection failed: {e}")
        with state.lock:
            state.shutdown = True
        return

    # Home position snapshot for translation reference
    xo, yo, zo, *_ = arm.position

    # Mode initialisation flags
    translation_ready = True
    orientation_ready = False
    preset_applied = False

    # Edge detection for gripper and preset gestures
    prev_right_closed = False
    prev_preset1 = False
    prev_preset2 = False

    print("[robot] Ready.")

    while arm.connected and arm.state != 4:
        with state.lock:
            shutdown = state.shutdown
            connected = state.connected
            move = state.move
            position = state.position.copy()
            translation_mode = state.translation_mode
            orientation_mode = state.orientation_mode
            tool_mode = state.tool_mode
            right_closed = state.right_closed
            preset1 = state.preset_1
            preset2 = state.preset_2

        if shutdown or not connected:
            break

        if not move:
            time.sleep(0.01)
            continue

        # --- Gripper toggle on right-fist rising edge ---
        if right_closed and not prev_right_closed:
            _toggle_gripper(arm, state)
        prev_right_closed = right_closed

        # --- Mode switch initialisation ---
        if translation_mode and not translation_ready:
            xo, yo, zo = _switch_to_translation(arm)
            translation_ready = True
            orientation_ready = False
            preset_applied = False
            print("[robot] Switched to translation mode.")

        elif orientation_mode and not orientation_ready:
            angles = _switch_to_orientation(arm)
            j4o, j5o, j6o = angles[3], angles[4], angles[5]
            translation_ready = False
            orientation_ready = True
            preset_applied = False
            print("[robot] Switched to orientation mode.")

        # --- Translation control (Mode 5 — Cartesian velocity) ---
        if translation_mode and translation_ready:
            xa, ya, za, *_ = arm.position
            vx = _scale_velocity(xo + position[0] - xa)
            vy = _scale_velocity(yo + position[1] - ya)
            vz = _scale_velocity(zo + position[2] - za)
            arm.vc_set_cartesian_velocity([vx, vy, vz, 0, 0, 0])

        # --- Orientation control (Mode 4 — joint velocity) ---
        elif orientation_mode and orientation_ready and not preset_applied:
            j4a, j5a, j6a = arm.angles[3], arm.angles[4], arm.angles[5]

            if not tool_mode:
                # J4 / J5 rotation
                vj4 = _scale_velocity(j4o + position[1] - j4a)
                vj5 = _scale_velocity(j5o + position[2] - j5a)
                arm.vc_set_joint_velocity([0, 0, 0, vj4, vj5, 0])
            else:
                # J6 rotation only
                vj6 = _scale_velocity(j6o + position[0] - j6a)
                arm.vc_set_joint_velocity([0, 0, 0, 0, 0, vj6])

            # --- Preset positions ---
            preset1_rising = preset1 and not prev_preset1
            preset2_rising = preset2 and not prev_preset2

            if preset1_rising or preset2_rising:
                arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0])
                time.sleep(1)
                arm.set_mode(0); arm.set_state(0)
                xa, ya, za, *_ = arm.position

                if preset1_rising:
                    arm.set_position(xa, ya, za, 180, 0, 0,
                                     speed=10, wait=True)
                else:
                    arm.set_position(xa, ya, za, 180, -90, 0,
                                     speed=10, wait=True)
                preset_applied = True

        prev_preset1 = preset1
        prev_preset2 = preset2

        time.sleep(0.01)

    _safe_shutdown(arm, state)