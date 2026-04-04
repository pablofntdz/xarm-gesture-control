#  xArm6 Gesture Control

Real-time control of a 6-DOF robotic arm using hand gesture recognition and RGB-D vision — no physical interface required.

---

## Overview

This project implements a markerless, hands-free control system for the **UFactory xArm6** robotic manipulator. Using a **Intel RealSense** depth camera and **MediaPipe Holistic**, the system tracks both hands in real time and maps gesture-based inputs to robot motion — including 3D translation, joint orientation, and gripper actuation.

The control loop runs across three concurrent threads:

- **Vision & Pose Estimation** — captures RGB-D frames, runs MediaPipe landmark detection, and computes 3D position vectors using homogeneous transformation matrices
- **Robot Control** — connects to the xArm6 controller over IP, switches between Cartesian velocity (translation) and joint velocity (orientation) modes, and enforces safety limits
- **Gesture Recognition** — classifies hand poses (open, closed, 1 finger, 2 fingers) and maps them to system commands

---

## Features

- **Two control modes** switchable via gesture:
  - *Translation mode* — move the end-effector in XYZ space
  - *Orientation mode* — rotate joints J4/J5 or J6 independently
- **Depth-corrected positioning** using a custom correction factor calibrated to the RealSense SR300
- **Gripper control** via right-hand close gesture (pneumatic gripper with digital I/O)
- **Predefined poses** triggered by finger count (1 or 2 fingers on both hands simultaneously)
- **Safe shutdown** — returns to home position and opens gripper on ESC or dual-hand close
- **Collision safety** configured with TCP tool model, reduced speed zone, and self-collision detection

---

## Hardware

| Component | Model |
|---|---|
| Robot arm | UFactory xArm6 |
| Depth camera | Intel RealSense SR300 |
| Gripper | Makeblock Robot Gripper (pneumatic) |

---

## Software & Dependencies

```
Python 3.x
opencv-python
mediapipe
numpy
pyrealsense2
xArm-Python-SDK
```

Install Python dependencies:

```bash
pip install opencv-python mediapipe numpy pyrealsense2
```

Clone and set up the xArm SDK:

```bash
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
```

Update the SDK path in `main.py` if needed:

```python
sys.path.append('/path/to/xArm-Python-SDK')
```

---

## Getting Started

1. Connect the Intel RealSense camera via USB
2. Connect the xArm6 controller to the same network
3. Update the robot IP address in `main.py`:

```python
ip = "192.168.1.XXX"
```

4. Run the system:

```bash
python main.py
```

---

## Gesture Reference

| Gesture | Hand | Action |
|---|---|---|
| Closed fist | Left | Set reference point / switch mode |
| Closed fist | Right | Toggle gripper open/close |
| 1 finger | Both | Move to preset pose 1 (roll=180°) |
| 2 fingers | Both | Move to preset pose 2 (roll=180°, pitch=−90°) |
| ESC key | — | Safe shutdown |
| Closed fist (both) | Both | Safe shutdown |

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│                    Main Process                  │
│                                                  │
│  Thread 1: Vision          Thread 3: Gestures    │
│  ┌──────────────┐          ┌──────────────────┐  │
│  │ RealSense    │          │ Hand open/close  │  │
│  │ RGB-D frames │          │ Finger count     │  │
│  │ MediaPipe    │          │ Bimanual combos  │  │
│  │ Holistic     │          └────────┬─────────┘  │
│  │ 3D transform │                   │             │
│  └──────┬───────┘                   │             │
│         │        Shared state       │             │
│         └──────────────────────────┤             │
│                                    ▼             │
│                      Thread 2: Robot Control     │
│                      ┌─────────────────────────┐ │
│                      │ xArm6 via IP            │ │
│                      │ Mode 4 (joint vel.)     │ │
│                      │ Mode 5 (Cartesian vel.) │ │
│                      │ Gripper I/O             │ │
│                      └─────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## Safety

The system applies several layers of safety before any motion:

- Reduced TCP speed limit (100 mm/s max)
- Reduced mode bounding box (custom Cartesian workspace)
- Collision sensitivity enabled
- Self-collision detection enabled
- TCP tool model defined as a cylinder (radius: 115 mm, height: 200 mm)
- Velocity is smoothly scaled by distance to target (dead zone + proportional ramp)

---

## Known Limitations

- Depth readings of `0` (occlusions or out-of-range) are handled by holding the last valid value
- Gesture classification is distance-based (tip vs. MCP), which can be sensitive to hand scale and camera distance
- Single-script architecture — refactoring into modules would improve testability and reuse

---

## License

MIT License. See `LICENSE` for details.