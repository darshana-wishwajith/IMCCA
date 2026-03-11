"""
config.py — Central Configuration for IMCCA v2
All tunable parameters in one place.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CameraConfig:
    """Camera capture settings."""
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    buffer_size: int = 1          # Minimal buffer = lowest latency


@dataclass
class PoseConfig:
    """MediaPipe Pose model settings."""
    model_complexity: int = 0      # 0 = Lite (fastest), 1 = Full, 2 = Heavy
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False


@dataclass
class SmoothingConfig:
    """Landmark smoothing parameters."""
    ema_alpha: float = 0.6         # 0 = full smooth, 1 = no smooth (raw)
    velocity_window: int = 4       # Frames to compute velocity over
    velocity_ema: float = 0.5      # Velocity smoothing


@dataclass
class ActionThresholds:
    """Thresholds for gesture detection (body-relative units)."""
    
    # === PUNCH DETECTION ===
    punch_velocity: float = 0.1  #0.5 # Wrist velocity threshold (body-widths/sec)
    punch_extension: float = 0.2      # Wrist must extend this far past elbow (body-relative)
    punch_cooldown: float = 0.10       # Seconds between punches
    
    # === KICK DETECTION ===
    kick_ankle_rise: float = 0.02      # Knee/ankle rise vs other leg (very small = easy trigger)
    kick_velocity: float = 0.3         # Ankle velocity threshold (low = small motions work)
    kick_leg_angle: float = 50    # Leg angle below this = kick (standing ≈ 170°, small raise ≈ 155°)
    kick_cooldown: float = 0.10        # Seconds between kicks

    # === BLOCK DETECTION ===
    block_wrist_distance: float = 0.40 # Both wrists must be within this dist of chest center
    block_arms_crossed: float = 0.20   # Wrists must cross body center by this much
    block_hold_time: float = 0.01     # Seconds to confirm block
    
    # === THROW DETECTION ===
    throw_velocity: float = 1.0        # Both wrists must spike forward together
    throw_sync_window: float = 0.1     # Both wrists must spike within this time window
    throw_cooldown: float = 0.40       # Seconds between throws
    
    # === MOVEMENT DETECTION (Spine Angle) ===
    move_angle_threshold: float = 85.0 # Degrees: spine-to-X angle below this = MOVE
                                         # 90° = perfectly upright, 75° = moderate lean
    
    # === JUMP / DUCK (Knee-Angle Based) ===
    duck_knee_angle: float = 170.0     # Degrees: knee angle below this = DUCK
                                  # Standing ≈ 170°, crouching < 140°
    jump_knee_angle: float = 160.0     # Knee angle must stay above this for jump
    jump_hip_velocity: float = 0.20    # Upward hip velocity required for jump
    jump_hip_rise: float = 0.03    # Hip must also rise this much (normalized)
    jump_cooldown: float = 0.50        # Seconds between jumps
    duck_hold_time: float = 0.10       # Seconds to confirm duck
    
    # === FLIP STANCE ===
    flip_rotation_speed: float = 2.5 # Shoulder rotation speed threshold
    flip_cooldown: float = 2       # Seconds between flips
    
    # === CHARACTER ASSIST ===
    assist_wrist_above_head: float = 0.10  # Wrists must be this far above nose
    assist_hold_time: float = 0.3         # Hold for this long to trigger
    assist_cooldown: float = 1.0          # Long cooldown to prevent spam
    
    # === FATAL BLOW (U + O) ===
    fatal_arms_spread: float = 1.4     # Wrists must be this × shoulder_width apart
    fatal_wrist_height: float = 0.15   # Wrists must be within this of shoulder Y
    fatal_hold_time: float = 0.05      # Seconds to confirm (very fast)
    fatal_cooldown: float = 2.0        # Long cooldown


# MK11 Keyboard Mapping — matches the game's control settings
MK11_KEYS: Dict[str, str] = {
    "FRONT_PUNCH":     "j",
    "BACK_PUNCH":      "i",
    "FRONT_KICK":      "k",
    "BACK_KICK":       "l",
    "THROW":           "space",
    "BLOCK":           "o",
    "INTERACT":        ";",
    "FLIP_STANCE":     "u",
    "JUMP":            "w",
    "DUCK":            "s",
    "MOVE_LEFT":       "a",
    "MOVE_RIGHT":      "d",
    "CHAR_ASSIST":     "p",
    "FATAL_BLOW":      "u+o",         # Simultaneous U + O
}


# Combo definitions: (sequence of actions within time window) -> (output key sequence)
# Each combo is: name, trigger_actions, time_window, output_keys_with_delays
@dataclass
class ComboDefinition:
    name: str
    trigger_sequence: List[str]         # e.g. ["FRONT_PUNCH", "FRONT_PUNCH", "FRONT_KICK"]
    time_window: float                   # Max seconds for full sequence
    output_keys: List[Tuple[str, float]] # (key, hold_duration) pairs to send


DEFAULT_COMBOS: List[ComboDefinition] = [
    ComboDefinition(
        name="Quick Combo 1",
        trigger_sequence=["FRONT_PUNCH", "FRONT_PUNCH", "FRONT_KICK"],
        time_window=1.5,
        output_keys=[("j", 0.05), ("j", 0.05), ("k", 0.05)],
    ),
    ComboDefinition(
        name="Quick Combo 2",
        trigger_sequence=["BACK_PUNCH", "BACK_PUNCH", "BACK_KICK"],
        time_window=1.5,
        output_keys=[("i", 0.05), ("i", 0.05), ("l", 0.05)],
    ),
    ComboDefinition(
        name="Upper combo",
        trigger_sequence=["FRONT_PUNCH", "BACK_PUNCH", "FRONT_KICK"],
        time_window=1.5,
        output_keys=[("j", 0.05), ("i", 0.05), ("k", 0.05)],
    ),
]


@dataclass
class UIConfig:
    """Display and UI settings."""
    window_name: str = "IMCCA — Immersive Motion Combat Arena"
    sidebar_width: int = 200
    show_skeleton: bool = True
    show_action_flash: bool = True      # Flash screen border on attacks
    action_flash_duration: float = 0.15 # Seconds
    font_scale: float = 0.5
