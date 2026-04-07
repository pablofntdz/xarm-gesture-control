"""
main.py
-------
Entry point for the xArm6 gesture control system.

Starts concurrent threads and waits for all of them to finish:
    - Vision thread: camera capture + MediaPipe + position computation
    - Gesture thread: hand pose classification
    - Robot thread: xArm6 velocity control (skipped in webcam mode)

Set WEBCAM_MODE = True to test vision and gesture recognition using a
standard webcam, without needing the RealSense camera or the xArm6.
"""

from threading import Thread

from state import SystemState
from vision import run_vision_thread
from gestures import run_gesture_thread

# --- Hardware mode ---
# True  → webcam only, no robot (for development/testing)
# False → full system: RealSense + xArm6
WEBCAM_MODE = True

if WEBCAM_MODE:
    from camera_webcam import DepthCamera
else:
    from camera import DepthCamera
    from robot import run_robot_thread


def main() -> None:
    state = SystemState()
    results_holder = [None]  # Shared MediaPipe results (vision → gestures)

    dc = DepthCamera()

    threads = [
        Thread(target=run_vision_thread,
               args=(dc, state, results_holder),
               daemon=True, name="vision"),
        Thread(target=run_gesture_thread,
               args=(state, results_holder),
               daemon=True, name="gestures"),
    ]

    if not WEBCAM_MODE:
        threads.append(
            Thread(target=run_robot_thread,
                   args=(state,),
                   daemon=True, name="robot")
        )

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("System stopped.")


if __name__ == "__main__":
    main()