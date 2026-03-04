# IMCCA - Immersive Motion Combat Control Arena

> Control fighting games like **Mortal Kombat 11** using real-time body and hand tracking through your webcam — no physical controller needed.

IMCCA transforms your webcam into a virtual Xbox 360 controller. Move your left hand to control movement (walk, jump, crouch), and use your right hand for attacks (punch, kick, block). The system uses MediaPipe for body/hand tracking and vgamepad to emulate an Xbox 360 controller with both analog joystick and D-PAD output.

---

## Features

### 🎮 Dual Control Modes

| Mode                | Description                                                                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SWIPE** (Default) | Digital D-PAD style. Open hand to activate, swipe in a direction to hold it. Fist releases all inputs. Best for fighting games.                    |
| **POSITION**        | Analog joystick style. Palm position relative to a neutral center maps proportionally to joystick values. Good for games that prefer analog input. |

Press **M** to toggle between modes during gameplay.

### ✋ Left Hand — Movement (Joystick / D-PAD)

- **Open hand** = tracking active
- **Swipe left/right/up/down** = movement in that direction (held until release)
- **Fist** = return to neutral (release all movement)
- Supports **4 cardinal directions** + **diagonals**
- Cardinal snapping when movement is strongly along one axis

### 🤜 Right Hand — Attacks (Swipe Mode Only)

| Gesture                     | Action                      |
| --------------------------- | --------------------------- |
| **Fist + fast motion**      | Punch (Xbox X button)       |
| **Open hand + fast motion** | Kick (Xbox A button)        |
| **Fist held still**         | Block (Xbox Right Shoulder) |

### 🎯 Auto-Calibration

- On startup, the system waits in **IDLE** mode
- Show **10 fingers** (both hands open) to **START** and auto-calibrate
- Calibration measures **shoulder width** to automatically adjust sensitivity based on your distance from the camera
- Show **V-sign** (index + middle finger) to **STOP** tracking

### 🛡️ Player Lock

- Uses MediaPipe Pose (single-person) to anchor hand tracking to the primary player's body
- Background people's hands are **automatically ignored** — only the hand closest to the primary player's wrist is tracked

### 📊 On-Screen UI

- **Sidebar** (right): LIVE/IDLE status, FPS, mode indicator, joystick X/Y values, direction output, sensitivity
- **Virtual Joystick** (bottom-left): Visual representation with UP/DN/L/R labels that highlight yellow when active
- **Palm Tracking Dot**: Yellow circle on left hand, green circle on right hand (swipe mode)

---

## Architecture

```
┌─────────────────────┐
│     body_skeleton.py │  ← Webcam capture + MediaPipe Pose/Hands inference
│     (Module 1)       │     Skeleton drawing, landmark extraction
└──────────┬──────────┘
           │ pose_landmarks, hand_landmarks
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  motion_detector.py │     │  swipe_detector.py   │
│  (Module 2a)        │     │  (Module 2b)         │
│  Proportional mode  │     │  Swipe/Digital mode  │
│  Analog joystick    │     │  D-PAD + Attacks     │
└──────────┬──────────┘     └──────────┬──────────┘
           │ joy_x, joy_y, actions      │
           ▼                            ▼
┌─────────────────────┐
│ input_controller.py │  ← Virtual Xbox 360 gamepad (vgamepad)
│ (Module 3)          │     Analog Joystick + D-PAD + Buttons
└──────────┬──────────┘
           │ USB HID
           ▼
┌─────────────────────┐
│   Game (MK11 etc.)  │
└─────────────────────┘
```

### File Overview

| File                  | Purpose                                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `main.py`             | Entry point. Orchestrates the main loop, UI rendering, mode toggling, and keyboard shortcuts.                              |
| `body_skeleton.py`    | Webcam capture and MediaPipe Pose/Hands inference. Provides skeleton drawing and landmark extraction.                      |
| `motion_detector.py`  | **Position mode** — Proportional joystick via relative palm tracking from a neutral center.                                |
| `swipe_detector.py`   | **Swipe mode** — Digital D-PAD via gesture-gated swipe detection. Includes right-hand attack detection (punch/kick/block). |
| `input_controller.py` | Virtual Xbox 360 controller via vgamepad. Maps joystick values to both analog stick and D-PAD simultaneously.              |

---

## Installation

### Prerequisites

- Python 3.9+
- Windows 10/11
- Webcam
- [ViGEmBus Driver](https://github.com/nefarius/ViGEmBus/releases) (required for vgamepad virtual controller)

### Setup

```bash
# Clone the repository
git clone https://github.com/darshana-wishwajith/IMCCA.git
cd IMCCA

# Create virtual environment
python -m venv imcca_venv
imcca_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package       | Version  | Purpose                               |
| ------------- | -------- | ------------------------------------- |
| opencv-python | 4.8.1.78 | Video capture and UI rendering        |
| mediapipe     | 0.9.3.0  | Pose and hand landmark detection      |
| numpy         | 1.24.4   | Numerical operations                  |
| vgamepad      | 0.1.0    | Virtual Xbox 360 controller emulation |

---

## Usage

```bash
python main.py
```

### Keyboard Shortcuts

| Key       | Action                                   |
| --------- | ---------------------------------------- |
| `ESC`     | Quit                                     |
| `M`       | Toggle between SWIPE and POSITION mode   |
| `+` / `=` | Increase sensitivity                     |
| `-`       | Decrease sensitivity                     |
| `R`       | Reset neutral center                     |
| `]`       | Increase attack sensitivity (Swipe mode) |
| `[`       | Decrease attack sensitivity (Swipe mode) |

### Gesture Controls

| Gesture                          | Action                          |
| -------------------------------- | ------------------------------- |
| Both hands open (10 fingers)     | START tracking + auto-calibrate |
| V-sign (index + middle finger)   | STOP tracking                   |
| Left hand open + swipe direction | Move character                  |
| Left hand fist                   | Release movement                |
| Right hand fist + fast motion    | Punch                           |
| Right hand open + fast motion    | Kick                            |
| Right hand fist held still       | Block                           |

---

## Configuration

Key parameters can be adjusted in real-time via keyboard shortcuts, or modified in the source code:

### Motion Detector (Position Mode)

- `JOY_DEADZONE`: Ignore jitter threshold (default: 0.08)
- `JOY_MAX_REACH`: Max displacement for full joystick range (auto-calibrated)

### Swipe Detector (Swipe Mode)

- `ACTIVATION_DISPLACEMENT`: Distance to trigger a swipe (auto-calibrated)
- `NEUTRAL_ZONE_RADIUS`: Distance to return to neutral (auto-calibrated)
- `ATTACK_DISP_THRESH`: Frame displacement to trigger attack (auto-calibrated)
- `FIST_CONFIRM_FRAMES`: Consecutive fist frames needed to confirm (default: 3)

---

## Game Setup (Mortal Kombat 11)

1. Install the [ViGEmBus driver](https://github.com/nefarius/ViGEmBus/releases) if not already installed.
2. Run `python main.py` and confirm the virtual controller appears in Windows game controller settings.
3. Launch Mortal Kombat 11.
4. In MK11 settings, ensure the controller is detected (it should appear as an Xbox 360 controller).
5. Stand in front of your webcam and show 10 fingers to start.
6. Use your left hand for movement and right hand for attacks!

---

## License

This project is developed for educational and research purposes.
