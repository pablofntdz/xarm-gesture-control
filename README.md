#  xArm6 Gesture Control

Real-time control of a 6-DOF robotic arm using hand gesture recognition and RGB-D vision — no physical interface required.

---

## Overview

This project implements a markerless, hands-free control system for the **UFactory xArm6** robotic manipulator. An **Intel RealSense** depth camera captures RGB-D frames which are processed by **MediaPipe HandLandmarker** to track both hands in real time. Gesture-based inputs are mapped to robot motion — including 3D translation, joint orientation, and gripper actuation.

The system runs across three concurrent threads communicating through a shared, lock-protected state object:

- **Vision** — captures RGB-D frames, runs MediaPipe landmark detection, and computes 3D position offset vectors using the RealSense SDK deprojection API
- **Gesture Recognition** — classifies hand poses (open, closed, 1 finger, 2 fingers) and maps them to system commands
- **Robot Control** — connects to the xArm6 over IP, switches between Cartesian velocity (translation) and joint velocity (orientation) modes, and enforces safety limits

The system also supports a **webcam-only mode** for developing and testing gesture recognition without any hardware.

<div align="center">
<img src="docs/robot_pickplace_video.gif" width="240"/>
</div>




---
## Features
- **Two control modes** switchable via gesture:
  - *Translation mode* — move the end-effector in XYZ space (xArm Mode 5)
  - *Orientation mode* — rotate joints J4/J5 or J6 independently (xArm Mode 4)
- **Accurate 3D positioning** using `rs2_deproject_pixel_to_point` with the camera's calibrated intrinsics
- **Gripper control** via right-hand close gesture (pneumatic gripper with digital I/O)
- **Predefined poses** triggered by bimanual finger count gestures
- **Safe shutdown** — stops motion, opens gripper, and returns to home position on ESC or dual-hand close
- **Collision safety** with TCP tool model, reduced speed zone, and self-collision detection
- **Webcam mode** for testing vision and gesture logic without hardware

---

## Hardware

| Component | Model |
|---|---|
| Robot arm | UFactory xArm6 |
| Depth camera | Intel RealSense SR300 |
| Gripper | Makeblock Robot Gripper (pneumatic) |

---

## Project Structure

```
src/
├── main.py            # Entry point — set WEBCAM_MODE here
├── state.py           # Shared state with threading lock
├── camera.py          # Intel RealSense wrapper
├── camera_webcam.py   # Standard webcam wrapper (no depth sensor)
├── vision.py          # MediaPipe detection + position computation
├── gestures.py        # Hand pose classification
├── robot.py           # xArm6 velocity control
└── download_model.py  # Downloads the MediaPipe .task model file
```

---

## Getting Started

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install opencv-python mediapipe numpy pyrealsense2
```

### Download the MediaPipe model

The HandLandmarker model is not included in the package — run this once:

```bash
uv run python src/download_model.py
```

This downloads `hand_landmarker.task` (~30 MB) to the project root.

---

## Running

### Webcam mode (no hardware required)

Set `WEBCAM_MODE = True` in `main.py` (default), then:

```bash
uv run python src/main.py
```

### Full system (RealSense + xArm6)

1. Set `WEBCAM_MODE = False` in `main.py`
2. Clone the xArm Python SDK:

```bash
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
```

3. Update the robot IP in `robot.py`:

```python
ROBOT_IP = "192.168.1.XXX"
```

4. Connect the RealSense camera via USB and the xArm6 controller to the same network, then:

```bash
uv run python src/main.py
```

---

## Gesture Reference

| Gesture | Hand | Action |
|---|---|---|
| Closed fist | Left | Set reference point / cycle control mode |
| Closed fist | Right | Toggle gripper open/close |
| 1 finger | Both simultaneously | Move to preset pose 1 (roll=180°) |
| 2 fingers | Both simultaneously | Move to preset pose 2 (roll=180°, pitch=−90°) |
| ESC key | — | Safe shutdown |
| Closed fist | Both simultaneously | Safe shutdown |

### Mode cycling (left fist)

Each left-fist close cycles through:
```
Translation → Orientation J4/J5 → Orientation J6 → Translation → ...
```

---

## System Architecture

```
┌───────────────────────────────────────────────────────┐
│                      Main Process                     │
│                                                       │
│  Thread 1: Vision            Thread 2: Gestures       │
│  ┌─────────────────┐         ┌─────────────────────┐  │
│  │ RealSense/Webcam│         │ Hand open/close     │  │
│  │ RGB-D frames    │         │ Finger count        │  │
│  │ MediaPipe       │─────────│ Bimanual combos     │  │
│  │ HandLandmarker  │         │ Gripper toggle      │  │
│  │ 3D transform    │         └──────────┬──────────┘  │
│  └────────┬────────┘                    │             │
│           │           SystemState       │             │
│           │           (threading.Lock)  │             │
│           └─────────────────────────────┤             │
│                                        ▼              │
│                          Thread 3: Robot Control      │
│                          ┌──────────────────────────┐ │
│                          │ xArm6 via IP             │ │
│                          │ Mode 5 (Cartesian vel.)  │ │
│                          │ Mode 4 (joint vel.)      │ │
│                          │ Gripper digital I/O      │ │
│                          └──────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

---

## Safety

The following safety measures are applied at startup before any motion:

- Reduced TCP speed limit (100 mm/s max)
- Cartesian workspace bounding box (reduced mode)
- Collision sensitivity enabled
- Self-collision detection enabled
- TCP tool model defined as a cylinder (radius: 115 mm, height: 200 mm)
- Velocity proportionally scaled by distance to target — dead zone below 5 mm

---

## Known Limitations

- Depth readings of `0` (occlusions or out-of-range pixels) are handled by holding the last valid value
- Gesture classification is distance-based (tip vs. MCP joint), which can be sensitive to hand scale and camera angle
- Without a depth sensor, position tracking is computed at a fixed depth (0.6 m) — translation accuracy is reduced in webcam mode

---

## License

MIT License. See `LICENSE` for details.