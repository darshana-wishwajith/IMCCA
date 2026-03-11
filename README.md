<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-BlazePose-00A67E?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" />
</p>

# 🎮 IMCCA — Immersive Motion Combat Control Arena

**A full-body motion controller for Mortal Kombat 11**, built for exhibitions and interactive installations where players fight using their real body movements captured through a standard webcam — no special hardware required.

> Stand in front of a camera, throw real punches, kicks, and blocks — and watch your character mirror your movements in MK11. No controllers, no VR headsets — just your body.

---

## 📖 Concept

IMCCA bridges the gap between the physical and digital worlds in fighting games. Using computer vision and pose estimation, it transforms a standard webcam into a full-body game controller.

The system captures the player's skeleton in real-time using Google's **MediaPipe BlazePose** model, then analyzes joint velocities, body angles, and limb positions to detect fighting game actions — punches, kicks, blocks, throws, jumps, ducks, and lateral movement. These detected actions are instantly translated into keyboard inputs that MK11 reads as standard game controls.

### Why This Approach?

- **Pose-Only Detection** — Uses only the BlazePose body model (33 landmarks), deliberately avoiding the hand tracking model. This sacrifices finger-level detail for a massive FPS boost, keeping the system responsive enough for real-time fighting game input.
- **Body-Relative Coordinates** — All thresholds are normalized against the player's own shoulder width and torso height, making the system work regardless of body size or distance from the camera.
- **Velocity + Position Hybrid** — Attacks aren't detected by position alone. The system requires both a velocity spike (you're actually moving fast) AND correct limb positioning, drastically reducing false triggers.

---

## 🏗️ Project Architecture

```
IMCCA/
├── main.py              → Entry point: camera loop, UI overlay, keyboard controls
├── config.py            → Central configuration: all tunable thresholds & key mappings
├── pose_tracker.py      → MediaPipe Pose wrapper: threaded camera, landmark smoothing, calibration
├── gesture_engine.py    → Core detection engine: velocity tracking, gesture classification, combos
├── input_sender.py      → Keyboard output: threaded tap/hold/combo key sender via pynput
├── requirements.txt     → Core Python dependencies
├── requirements-lock.txt→ Full pinned dependency lock file
└── README.md
```

### Module Dependency Flow

```
main.py
 ├─→ config.py            (CameraConfig, PoseConfig, ActionThresholds, MK11_KEYS, ...)
 ├─→ pose_tracker.py      (ThreadedCamera, PoseTracker)
 │    └─→ config.py
 ├─→ gesture_engine.py    (VelocityTracker, GestureEngine)
 │    ├─→ config.py
 │    └─→ pose_tracker.py
 └─→ input_sender.py      (InputSender)
      └─→ config.py
```

### Module Responsibilities

| Module                  | Role                                                                                                                                                                                                                                                             | Key Classes                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **`main.py`**           | Application entry point. Runs the main camera loop, handles UI rendering (status bar, action display, virtual joystick, help text), and connects all modules together.                                                                                           | `draw_ui()`, `main()`                                                                              |
| **`config.py`**         | Centralized configuration. Every tunable parameter lives here — camera settings, pose model config, smoothing params, gesture thresholds, keyboard mappings, and combo definitions.                                                                              | `CameraConfig`, `PoseConfig`, `SmoothingConfig`, `ActionThresholds`, `UIConfig`, `ComboDefinition` |
| **`pose_tracker.py`**   | Handles camera capture and pose estimation. Runs camera reads in a background thread for zero-latency frame access. Processes frames through MediaPipe Pose, applies EMA smoothing, and provides body-relative coordinate conversion.                            | `ThreadedCamera`, `PoseTracker`                                                                    |
| **`gesture_engine.py`** | The brain of the system. Tracks joint velocities via EMA-smoothed history windows. Detects all game actions using velocity spikes, positional checks, angular measurements, and timed holds. Also handles combo sequence matching.                               | `VelocityTracker`, `GestureEngine`                                                                 |
| **`input_sender.py`**   | Translates detected actions into keyboard input. Uses pynput for direct key simulation. Tap actions (punches, kicks) go through a background thread queue. Hold actions (block, duck, move) stay pressed until released. Supports simultaneous multi-key combos. | `InputSender`                                                                                      |

---

## 🎯 Motion-to-Action Mapping

| Body Motion                     | Game Action      | Keyboard Key | Detection Method                          |
| ------------------------------- | ---------------- | ------------ | ----------------------------------------- |
| Right hand punch forward        | Front Punch      | `J`          | Wrist velocity spike + arm extension      |
| Left hand punch forward         | Back Punch       | `I`          | Wrist velocity spike + arm extension      |
| Right leg kick                  | Front Kick       | `K`          | Ankle velocity + knee rise / leg angle    |
| Left leg kick                   | Back Kick        | `L`          | Ankle velocity + knee rise / leg angle    |
| Cross arms on chest             | Block            | `O`          | Both wrists close to chest center (hold)  |
| Both hands thrust forward       | Throw            | `Space`      | Both wrists high velocity, same direction |
| Lean body left                  | Move Left        | `A`          | Spine midline angle deviation             |
| Lean body right                 | Move Right       | `D`          | Spine midline angle deviation             |
| Jump upward                     | Jump             | `W`          | Hip upward velocity + knees straight      |
| Crouch / squat                  | Duck             | `S`          | Average knee angle below threshold        |
| Quick body twist                | Flip Stance      | `U`          | Shoulder Z-depth differential             |
| Both hands above head (hold)    | Character Assist | `P`          | Wrists above nose for sustained duration  |
| Arms spread wide (T-pose, hold) | Fatal Blow       | `U+O`        | Wrist separation > 1.4× shoulder width    |

### Combo System

The engine tracks recent action sequences and matches them against predefined combo patterns within a time window:

| Combo Name    | Trigger Sequence                       | Time Window |
| ------------- | -------------------------------------- | ----------- |
| Quick Combo 1 | Front Punch → Front Punch → Front Kick | 1.5s        |
| Quick Combo 2 | Back Punch → Back Punch → Back Kick    | 1.5s        |
| Upper Combo   | Front Punch → Back Punch → Front Kick  | 1.5s        |

---

## 🛠️ Tech Stack

| Technology    | Purpose                                        | Version  |
| ------------- | ---------------------------------------------- | -------- |
| **Python**    | Core language                                  | 3.8+     |
| **MediaPipe** | Pose estimation (BlazePose model)              | 0.10.11  |
| **OpenCV**    | Camera capture, image processing, UI rendering | 4.8.1.78 |
| **NumPy**     | Numerical operations, landmark smoothing       | 1.24.4   |
| **pynput**    | Keyboard input simulation (tap, hold, combo)   | 1.7.6    |

### Why These Choices?

- **MediaPipe over YOLO/OpenPose** — Runs entirely on CPU with TFLite, gives 33 body landmarks at 30+ FPS on modest hardware, and includes built-in landmark smoothing.
- **pynput over PyDirectInput** — Supports both tap and hold key behaviors with simultaneous multi-key presses, essential for moves like Fatal Blow (`U+O`).
- **OpenCV for UI** — Keeps everything in a single rendering pipeline. The camera frame, skeleton overlay, action feedback, and virtual joystick are all drawn directly onto the frame.

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8 – 3.11** (MediaPipe requires Python ≤ 3.11)
- **Windows 10/11** (uses `cv2.CAP_DSHOW` for DirectShow camera access)
- **Webcam** (built-in or USB, 640×480 minimum recommended)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/darshana-wishwajith/IMCCA.git
cd IMCCA

# 2. Create a virtual environment
python -m venv imcca_venv

# 3. Activate the virtual environment
imcca_venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python main.py
```

### First-Time Usage

1. Stand **2–3 meters** from the camera so your full body (head to ankles) is visible.
2. The system **auto-calibrates** on the first frame where it sees your body.
3. Once **6 out of 8 key body points** are visible, LIVE mode activates automatically.
4. Start fighting! Throw punches, kicks, and blocks naturally.

---

## 🎮 In-App Controls

| Key       | Action                                         |
| --------- | ---------------------------------------------- |
| `ESC`     | Quit the application                           |
| `R`       | Recalibrate body measurements                  |
| `+` / `=` | Increase sensitivity (lower attack thresholds) |
| `-`       | Decrease sensitivity (raise attack thresholds) |

---

## 📐 How Detection Works

### Calibration

On the first frame (or when pressing `R`), the system measures:

- **Shoulder width** — Used as the primary scaling unit for all thresholds
- **Torso height** — Shoulder midpoint to hip midpoint distance
- **Standing hip Y** — Baseline hip position for jump detection
- **Body center X** — For body-relative coordinate conversion

All gesture thresholds are then scaled proportionally, making the system **body-size invariant**.

### Velocity Tracking

Each key joint (wrists, ankles, hip center) has a dedicated `VelocityTracker` that:

1. Maintains a sliding window of recent positions (default: 4 frames)
2. Computes raw velocity from oldest-to-newest displacement
3. Applies EMA (Exponential Moving Average) smoothing to reduce noise

### Detection Priority

Actions are checked in a strict priority order each frame to prevent conflicts:

```
1. Fatal Blow   (highest — both arms spread wide)
2. Block        (arms crossed on chest, suppresses punches)
3. Throw        (both hands fast in same direction)
4. Punches      (single wrist velocity + extension)
5. Kicks        (ankle velocity + leg position)
6. Movement     (spine angle lean)
7. Jump / Duck  (knee angle + hip velocity)
8. Flip Stance  (shoulder Z-depth rotation)
9. Char Assist  (both hands above head, timed hold)
```

---

## ⚠️ Limitations

- **Windows-only** — Uses `cv2.CAP_DSHOW` (DirectShow) for camera access and pynput for keyboard simulation. Linux/macOS would require alternative backends.
- **Single player** — Tracks only one body. If multiple people are in frame, detection may become unreliable.
- **Lighting sensitive** — MediaPipe Pose performance degrades in very low light or with strong backlighting.
- **Camera distance** — Player must be 1.5–4 meters from camera. Too close cuts off limbs; too far reduces landmark precision.
- **No diagonal movement** — Movement detection uses spine angle lean, which naturally supports only left/right. Jump and duck are separate from the movement axis.
- **CPU-bound** — MediaPipe Pose runs on CPU via TFLite. On older hardware, FPS may drop below the 25+ target, increasing input latency.
- **Game window focus** — pynput sends keyboard events globally. MK11 must be the focused window to receive inputs.
- **Ankle visibility** — The full-body detection requires ankles to be visible. Furniture, desks, or tight framing can cause frequent LIVE/IDLE toggling.

---

## 🔮 Future Improvements

- **GPU Acceleration** — Migrate to MediaPipe's GPU pipeline or use ONNX Runtime with CUDA for higher FPS on systems with dedicated GPUs.
- **Multi-Player Support** — Track two players simultaneously with split-screen skeleton isolation for PvP exhibition matches.
- **Adaptive Thresholds** — Machine learning-based threshold calibration that adapts to each player's movement style and speed over time.
- **Improved Movement** — Implement full 8-directional movement using a combination of hip offset and shoulder facing angle.
- **Visual Feedback Enhancements** — Add particle effects on attack detection, health bar sync from game state via screen capture, and a training mode overlay.
- **Cross-Platform Support** — Abstract the camera backend and input simulation layers to support Linux (`v4l2`) and macOS (`AVFoundation`).
- **Web Dashboard** — Real-time browser-based monitoring showing detection confidence, FPS graphs, action history, and remote threshold tuning.
- **Game Agnosticism** — Configurable action-to-key mapping profiles for other fighting games (Street Fighter 6, Tekken 8, etc.).
- **Gesture Recording** — Record and replay gesture sessions for debugging, training data collection, or demo playback.
- **Latency Optimization** — Explore frame skipping strategies, half-resolution pose estimation, and predictive input to reduce end-to-end latency below 50ms.

---

## 📄 License

This project is developed for educational and exhibition purposes.

---

<p align="center">
  <b>Built with 💪 for Extru 2026</b>
</p>
