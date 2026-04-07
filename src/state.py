"""
Shared state between threads.

Instead of plain global variables, all inter-thread communication lives
in a single SystemState instance protected by a threading.Lock.
This eliminates race conditions and makes data flow explicit.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SystemState:
    """
    Central shared state for the gesture-control system.

    All three threads (vision, robot, gesture) read and write to this
    object. Access is protected by `lock` to avoid race conditions.

    Usage:
        state = SystemState()
        with state.lock:
            state.connected = True
    """

    lock: threading.Lock = field(default_factory=threading.Lock)

    # --- Lifecycle flags ---
    connected: bool = False   # True once the camera produces its first frame
    move: bool = False        # True once a reference point has been set
    shutdown: bool = False    # Set to True to trigger a safe shutdown

    # --- Position vector (metres, camera frame) ---
    # Pi[0] = X offset, Pi[1] = Y offset, Pi[2] = Z offset
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # --- Gesture state ---
    left_closed: bool = False    # Left fist detected
    right_closed: bool = False   # Right fist detected

    # --- Control mode ---
    translation_mode: bool = True   # Cartesian velocity control (Mode 5)
    orientation_mode: bool = False  # Joint velocity control (Mode 4)
    tool_mode: bool = False         # J6 rotation sub-mode within orientation

    # --- Preset positions (triggered by finger count) ---
    preset_1: bool = False   # 1 finger both hands → roll=180°
    preset_2: bool = False   # 2 fingers both hands → roll=180°, pitch=-90°

    # --- Gripper state ---
    gripper_open: bool = True   # True = gripper open

    # --- Frame dimensions (set by vision thread on first frame) ---
    frame_width: int = 0
    frame_height: int = 0