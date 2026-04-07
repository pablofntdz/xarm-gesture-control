"""
gestures.py
-----------
Hand gesture recognition thread — MediaPipe Tasks API (0.10+).

Reads the results_holder dict written by vision.py:
    results_holder[0] = {"left": landmarks | None, "right": landmarks | None}

Each landmarks value is a list of 21 NormalizedLandmark objects.
Classification is distance-based: a finger is extended when its tip
is farther from the wrist than its MCP joint.
"""

import time
import numpy as np

from state import SystemState

# Landmark indices
WRIST            = 0
INDEX_FINGER_MCP = 5
MIDDLE_FINGER_MCP = 9
RING_FINGER_MCP  = 13
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP  = 16
PINKY_TIP        = 20
PINKY_MCP        = 17


def _finger_extended(tip_x: float, tip_y: float,
                     mcp_x: float, mcp_y: float,
                     wrist_x: float, wrist_y: float) -> bool:
    """
    Return True if a finger tip is farther from the wrist than its MCP joint.
    Uses Euclidean distance in 2D normalised coordinates.
    """
    tip   = np.array([tip_x,   tip_y])
    mcp   = np.array([mcp_x,   mcp_y])
    wrist = np.array([wrist_x, wrist_y])
    return float(np.linalg.norm(tip - wrist)) >= float(np.linalg.norm(mcp - wrist))


def _count_extended_fingers(landmarks: list) -> int:
    """
    Count how many of the 4 fingers (index→pinky) are extended.

    Args:
        landmarks: list of 21 NormalizedLandmark objects from HandLandmarker.

    Returns:
        Integer 0–3 representing extended finger count (thumb excluded).
    """
    wrist_x = landmarks[WRIST].x
    wrist_y = landmarks[WRIST].y

    finger_pairs = [
        (INDEX_FINGER_TIP,  INDEX_FINGER_MCP),
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        (RING_FINGER_TIP,   RING_FINGER_MCP),
    ]

    count = 0
    for tip_id, mcp_id in finger_pairs:
        if _finger_extended(
            landmarks[tip_id].x,  landmarks[tip_id].y,
            landmarks[mcp_id].x,  landmarks[mcp_id].y,
            wrist_x, wrist_y,
        ):
            count += 1

    return count


def run_gesture_thread(state: SystemState, results_holder: list) -> None:
    """
    Gesture recognition loop (runs in its own thread).

    Reads results_holder[0] written by the vision thread.

    Args:
        state: shared SystemState instance.
        results_holder: single-element list containing a dict
                        {'left': landmarks|None, 'right': landmarks|None}.
    """
    left_preset1  = False
    left_preset2  = False
    right_preset1 = False
    right_preset2 = False

    while True:
        data = results_holder[0]

        if data is None:
            time.sleep(0.01)
            if state.shutdown:
                break
            continue

        left_lm  = data.get("left")
        right_lm = data.get("right")

        left_closed  = False
        right_closed = False

        # --- Left hand ---
        if left_lm is not None:
            count = _count_extended_fingers(left_lm)
            if count == 0:
                left_closed  = True
                left_preset1 = False
                left_preset2 = False
            elif count == 1:
                left_preset1 = True
                left_preset2 = False
            elif count == 2:
                left_preset1 = False
                left_preset2 = True
            else:
                left_preset1 = False
                left_preset2 = False

        # --- Right hand ---
        if right_lm is not None:
            count = _count_extended_fingers(right_lm)
            if count == 0:
                right_closed  = True
                right_preset1 = False
                right_preset2 = False
            elif count == 1:
                right_preset1 = True
                right_preset2 = False
            elif count == 2:
                right_preset1 = False
                right_preset2 = True
            else:
                right_preset1 = False
                right_preset2 = False

        # --- Write to shared state ---
        with state.lock:
            state.left_closed  = left_closed
            state.right_closed = right_closed

            # Both hands closed simultaneously → safe shutdown
            if left_closed and right_closed:
                state.shutdown = True

            # Bimanual preset triggers (only active in orientation mode)
            if state.orientation_mode:
                state.preset_1 = left_preset1 and right_preset1
                state.preset_2 = left_preset2 and right_preset2

        time.sleep(0.01)  # ~100 Hz cap — avoids busy-waiting

        if state.shutdown:
            break