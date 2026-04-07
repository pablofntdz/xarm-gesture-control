"""
Vision and pose estimation thread — MediaPipe Tasks API (0.10+).

Uses HandLandmarker in VIDEO mode (synchronous, per-frame) which is
the right choice for a thread that already controls its own loop.

Key differences from the legacy Holistic API:
    - Result is HandLandmarkerResult, not a holistic result object.
    - Landmarks are in result.hand_landmarks[i] (list of hands).
    - Handedness is in result.handedness[i][0].category_name → "Left"/"Right".
    - Requires a .task model file downloaded separately (see download_model.py).
    - Timestamps must be monotonically increasing (we use frame counter × ms).
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from camera import DepthCamera  # also accepts camera_webcam.DepthCamera
from state import SystemState

# Landmark indices (same as legacy API)
WRIST             = 0
INDEX_FINGER_MCP  = 5
MIDDLE_FINGER_MCP = 9
RING_FINGER_MCP   = 13
PINKY_MCP         = 17
INDEX_FINGER_TIP  = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP   = 16
PINKY_TIP         = 20

MODEL_PATH = "hand_landmarker.task"

def _build_landmarker() -> mp_vision.HandLandmarker:
    """
    Create a HandLandmarker in VIDEO mode.

    VIDEO mode processes frames synchronously and uses tracking between
    frames to reduce latency — ideal for our thread-controlled loop.
    """
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _get_hand(result: mp_vision.HandLandmarkerResult,
              side: str) -> list | None:
    """
    Extract landmarks for the requested hand side from a result.

    Args:
        result: HandLandmarkerResult from detect_for_video().
        side: "Left" or "Right".

    Returns:
        List of 21 NormalizedLandmark objects, or None if not detected.

    Note: MediaPipe reports handedness assuming a mirrored (selfie) image.
    Since we flip the display but NOT the detection input, "Left" in the
    result corresponds to the user's left hand.
    """
    for i, handedness_list in enumerate(result.handedness):
        if handedness_list[0].category_name == side:
            return result.hand_landmarks[i]
    return None


def _draw_hud(frame: np.ndarray, state: SystemState) -> None:
    """Render mode labels, gripper status, and position offset onto the display frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    if state.translation_mode:
        cv2.putText(frame, "Mode: Translation",
                    (50, h - 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif state.orientation_mode:
        label = "Mode: Orientation - J6" if state.tool_mode \
                else "Mode: Orientation - J4/J5"
        cv2.putText(frame, label,
                    (50, h - 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    gripper_color = (0, 255, 0) if state.gripper_open else (0, 0, 255)
    gripper_label = "Gripper: OPEN" if state.gripper_open else "Gripper: CLOSED"
    cv2.putText(frame, gripper_label,
                (50, 50), font, 1, gripper_color, 2, cv2.LINE_AA)

    # Position offset from reference point (mm)
    pos = state.position
    cv2.putText(frame, f"dX: {pos[0]:+.1f} mm",
                (50, 100), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"dY: {pos[1]:+.1f} mm",
                (50, 130), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"dZ: {pos[2]:+.1f} mm",
                (50, 160), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)


def run_vision_thread(dc: DepthCamera, state: SystemState,
                      results_holder: list) -> None:
    """
    Main vision loop (runs in its own thread).

    Args:
        dc: initialised DepthCamera (or DepthCamera_webcam) instance.
        state: shared SystemState.
        results_holder: single-element list; this thread writes a dict
                        {'left': landmarks|None, 'right': landmarks|None}
                        for the gesture thread to read.
    """
    ref_point_3d: tuple[float, float, float] | None = None  # 3D reference in camera space
    reference_pt: tuple[int, int] | None = None            # 2D pixel to redraw on HUD
    mode_initialised = False
    last_valid_z     = 0.0
    prev_left_closed = False
    frame_index      = 0

    with _build_landmarker() as landmarker:
        while True:
            ret, color_frame, depth_frame = dc.get_frame()
            if not ret:
                break

            h, w = color_frame.shape[:2]
            with state.lock:
                state.frame_height = h
                state.frame_width  = w
                state.connected    = True

            # MediaPipe Tasks expects RGB mp.Image
            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=frame_rgb)

            # Timestamp must be monotonically increasing (milliseconds)
            timestamp_ms = frame_index * 33   # ~30 fps
            frame_index += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Extract per-side landmarks and share with gesture thread
            left_lm  = _get_hand(result, "Left")
            right_lm = _get_hand(result, "Right")
            results_holder[0] = {"left": left_lm, "right": right_lm}

            if right_lm is not None:
                # Control point: middle finger MCP joint
                cx = right_lm[MIDDLE_FINGER_MCP].x * w
                cy = right_lm[MIDDLE_FINGER_MCP].y * h
                control_pt = (int(cx), int(cy))

                z = depth_frame.get_distance(int(cx), int(cy))

                with state.lock:
                    left_closed = state.left_closed
                    orientation = state.orientation_mode

                left_rising      = left_closed and not prev_left_closed
                prev_left_closed = left_closed

                # --- Set / update reference point on left-fist rising edge ---
                if left_rising and z != 0:
                    if not mode_initialised:
                        mode_initialised = True
                        with state.lock:
                            state.translation_mode = True
                            state.orientation_mode = False
                            state.tool_mode        = False
                    else:
                        with state.lock:
                            if state.orientation_mode and not state.tool_mode:
                                state.tool_mode = True
                            elif state.orientation_mode and state.tool_mode:
                                state.tool_mode        = False
                                state.translation_mode = True
                                state.orientation_mode = False
                            else:
                                state.translation_mode = False
                                state.orientation_mode = True

                    # Store 3D reference point in camera space using SDK deprojection
                    ref_point_3d = dc.deproject(int(cx), int(cy), z)
                    reference_pt = control_pt
                    cv2.circle(color_frame, control_pt, 5, (255, 0, 0), -1)

                # --- Compute position offset vector ---
                if ref_point_3d is not None:
                    with state.lock:
                        state.move = True

                    if z != 0:
                        last_valid_z = z
                    else:
                        z = last_valid_z

                    # Deproject current pixel to 3D, then subtract reference → offset in metres
                    curr_x, curr_y, curr_z = dc.deproject(int(cx), int(cy), z)
                    position = np.array([
                        (curr_x - ref_point_3d[0]) * 1000,  # mm
                        (curr_y - ref_point_3d[1]) * 1000,  # mm
                        (curr_z - ref_point_3d[2]) * 1000,  # mm
                    ])

                    with state.lock:
                        state.position = position

                cv2.circle(color_frame, control_pt, 3, (0, 255, 0), 3)

            # Redraw fixed reference point every frame so it stays visible
            if reference_pt is not None:
                cv2.circle(color_frame, reference_pt, 6, (255, 255, 255), 2)

            # Flip for natural mirror view and draw HUD
            color_frame = cv2.flip(color_frame, 1)
            if mode_initialised:
                _draw_hud(color_frame, state)

            cv2.imshow("xArm Gesture Control", color_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                with state.lock:
                    state.shutdown = True
                break

            with state.lock:
                if state.shutdown:
                    break

    dc.release()
    cv2.destroyAllWindows()
    with state.lock:
        state.connected = False
        state.shutdown  = True