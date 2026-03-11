"""
gesture_engine.py — Full Body Gesture Detection Engine
IMCCA v2

Detects all MK11 actions using ONLY MediaPipe Pose landmarks.
No hand model needed — uses body joint velocities and positions.

Detection methods:
- Velocity spikes → Punches, Kicks, Throw
- Position zones → Block, Jump, Duck, Movement
- Timed holds → Block confirm, Duck confirm
- Combos → Timed action sequences
"""

import time
import math
from collections import deque
from config import ActionThresholds, DEFAULT_COMBOS, ComboDefinition
from pose_tracker import PoseTracker as PT
from typing import List, Dict, Tuple, Optional


class VelocityTracker:
    """Tracks velocity of a single landmark over a time window."""
    
    def __init__(self, window_size=4, ema_alpha=0.5):
        self._history = deque(maxlen=window_size + 1)
        self._ema_vx = 0.0
        self._ema_vy = 0.0
        self._alpha = ema_alpha
    
    def update(self, x, y, t):
        """Add a new position sample."""
        self._history.append((x, y, t))
    
    def get_velocity(self):
        """Get smoothed velocity (units/sec) in X and Y."""
        if len(self._history) < 2:
            return 0.0, 0.0
        
        old = self._history[0]
        new = self._history[-1]
        dt = new[2] - old[2]
        
        if dt < 0.001:
            return self._ema_vx, self._ema_vy
        
        raw_vx = (new[0] - old[0]) / dt
        raw_vy = (new[1] - old[1]) / dt
        
        # EMA smooth
        a = self._alpha
        self._ema_vx = a * raw_vx + (1 - a) * self._ema_vx
        self._ema_vy = a * raw_vy + (1 - a) * self._ema_vy
        
        return self._ema_vx, self._ema_vy
    
    def get_speed(self):
        """Get scalar speed."""
        vx, vy = self.get_velocity()
        return (vx**2 + vy**2)**0.5
    
    def reset(self):
        self._history.clear()
        self._ema_vx = 0.0
        self._ema_vy = 0.0


class GestureEngine:
    """
    Detects all game actions from pose landmarks.
    Returns a list of active actions each frame.
    """
    
    def __init__(self, thresholds=None, combos=None):
        self.thresh = thresholds or ActionThresholds()
        self.combos = combos or DEFAULT_COMBOS
        
        # Velocity trackers for key joints
        self._vel = {
            "r_wrist": VelocityTracker(),
            "l_wrist": VelocityTracker(),
            "r_ankle": VelocityTracker(),
            "l_ankle": VelocityTracker(),
            "hip_center": VelocityTracker(),
        }
        
        # Cooldown timestamps {action_name: next_allowed_time}
        self._cooldowns: Dict[str, float] = {}
        
        # State tracking
        self._block_start: Optional[float] = None
        self._duck_start: Optional[float] = None
        self._assist_start: Optional[float] = None
        self._is_blocking = False
        self._is_ducking = False
        self._fatal_start: Optional[float] = None
        
        # Movement state (for hold behavior)
        self._move_dir = 0   # -1 left, 0 center, +1 right
        self._jump_active = False
        
        # Combo tracking
        self._action_log: deque = deque(maxlen=20)  # (action, timestamp)
        self._last_combo_time = 0.0
        
        # Calibration state
        self._calibrated = False
        self._shoulder_width = 0.15  # Default until calibrated
        
        # Start/Stop gesture state
        self._gesture_history: deque = deque(maxlen=10)
        
        # For UI display
        self.active_actions: List[str] = []
        self.last_attack: Optional[str] = None
        self.last_attack_time: float = 0
        self.move_x: float = 0.0  # For joystick display
        self.move_y: float = 0.0
        self.spine_angle: float = 90.0  # Current spine angle for UI (90 = upright)
    
    def calibrate(self, landmarks, tracker: 'PoseTracker'):
        """Calibrate using current body measurements."""
        if tracker.calibrate(landmarks):
            self._shoulder_width = tracker.shoulder_width
            self._calibrated = True
            
            # Reset all state
            for v in self._vel.values():
                v.reset()
            self._cooldowns.clear()
            self._block_start = None
            self._duck_start = None
            self._assist_start = None
            self._is_blocking = False
            self._is_ducking = False
            self._fatal_start = None
            self._move_dir = 0
            self._jump_active = False
            self._action_log.clear()
            
            # Scale thresholds based on body size
            sw = self._shoulder_width
            self.thresh.punch_velocity = max(0.8, sw * 8.0)
            self.thresh.kick_velocity = max(0.5, sw * 5.0)
            self.thresh.throw_velocity = max(1.0, sw * 10.0)
            
            print(f">>> [ENGINE] Thresholds scaled: punch_vel={self.thresh.punch_velocity:.2f} "
                  f"kick_vel={self.thresh.kick_velocity:.2f} "
                  f"move_angle={self.thresh.move_angle_threshold:.0f}°")
            return True
        return False
    
    def detect(self, landmarks, tracker: 'PoseTracker') -> List[str]:
        """
        Main detection method. Call every frame.
        Returns list of active action names.
        """
        if not landmarks or not self._calibrated:
            self.active_actions = []
            self.move_x = 0.0
            self.move_y = 0.0
            return []
        
        now = time.time()
        actions = []
        
        # --- UPDATE VELOCITY TRACKERS ---
        self._update_velocities(landmarks, now)
        
        # --- DETECT ATTACKS (highest priority) ---
        # Priority: Fatal Blow > Block > Throw > Punches > Kicks
        
        # FATAL BLOW (arms spread wide) — check FIRST, highest priority
        fatal = None # Temporarily disabled: self._detect_fatal_blow(landmarks, now)
        if fatal:
            actions.append("FATAL_BLOW")
        
        is_blocking = False
        if not fatal:
            is_blocking = self._detect_block(landmarks, now)
            if is_blocking:
                actions.append("BLOCK")
        
        if not fatal and not is_blocking:
            throw = self._detect_throw(landmarks, now)
            if throw:
                actions.append("THROW")
            else:
                punch = self._detect_punches(landmarks, tracker, now)
                if punch:
                    actions.append(punch)
                
                kick = self._detect_kicks(landmarks, tracker, now)
                if kick:
                    actions.append(kick)
        
        # --- DETECT MOVEMENT (lower priority) ---
        move_action, mx, my = self._detect_movement(landmarks, tracker, now)
        self.move_x = mx
        self.move_y = my
        if move_action:
            actions.append(move_action)
        
        # --- DETECT JUMP / DUCK ---
        jump_duck = self._detect_jump_duck(landmarks, tracker, now)
        if jump_duck:
            actions.append(jump_duck)
        
        # --- DETECT SPECIAL ACTIONS ---
        flip = self._detect_flip_stance(landmarks, now)
        if flip:
            actions.append(flip)
        
        assist = self._detect_char_assist(landmarks, now)
        if assist:
            actions.append(assist)
        
        # --- COMBO DETECTION ---
        # Log attack actions for combo tracking
        for a in actions:
            if a in ("FRONT_PUNCH", "BACK_PUNCH", "FRONT_KICK", "BACK_KICK", "THROW"):
                self._action_log.append((a, now))
        
        combo = self._detect_combo(now)
        if combo:
            actions.append(f"COMBO:{combo}")
        
        self.active_actions = actions
        return actions
    
    def _can_fire(self, action: str, now: float) -> bool:
        """Check if an action is off cooldown."""
        return now >= self._cooldowns.get(action, 0)
    
    def _set_cooldown(self, action: str, now: float, duration: float):
        """Set cooldown for an action."""
        self._cooldowns[action] = now + duration
    
    # ─── VELOCITY UPDATES ────────────────────────────────────────
    
    def _update_velocities(self, lms, now):
        """Update velocity trackers for key joints."""
        joints = {
            "r_wrist": PT.RIGHT_WRIST,
            "l_wrist": PT.LEFT_WRIST,
            "r_ankle": PT.RIGHT_ANKLE,
            "l_ankle": PT.LEFT_ANKLE,
        }
        
        for name, idx in joints.items():
            if idx in lms:
                self._vel[name].update(lms[idx][0], lms[idx][1], now)
        
        # Hip center
        lh = lms.get(PT.LEFT_HIP)
        rh = lms.get(PT.RIGHT_HIP)
        if lh and rh:
            hx = (lh[0] + rh[0]) / 2
            hy = (lh[1] + rh[1]) / 2
            self._vel["hip_center"].update(hx, hy, now)
    
    # ─── PUNCH DETECTION ─────────────────────────────────────────
    
    def _detect_punches(self, lms, tracker, now) -> Optional[str]:
        """
        Detect punches based on wrist velocity spike + extension.
        Right hand punch = FRONT_PUNCH (J key)
        Left hand punch = BACK_PUNCH (I key)
        """
        # Check right wrist (user's right hand) → FRONT PUNCH
        if self._can_fire("FRONT_PUNCH", now):
            r_speed = self._vel["r_wrist"].get_speed()
            r_extended = self._is_arm_extended(lms, "right", tracker)
            
            if r_speed > self.thresh.punch_velocity and r_extended:
                self._set_cooldown("FRONT_PUNCH", now, self.thresh.punch_cooldown)
                self.last_attack = "FRONT_PUNCH"
                self.last_attack_time = now
                return "FRONT_PUNCH"
        
        # Check left wrist (user's left hand) → BACK PUNCH
        if self._can_fire("BACK_PUNCH", now):
            l_speed = self._vel["l_wrist"].get_speed()
            l_extended = self._is_arm_extended(lms, "left", tracker)
            
            if l_speed > self.thresh.punch_velocity and l_extended:
                self._set_cooldown("BACK_PUNCH", now, self.thresh.punch_cooldown)
                self.last_attack = "BACK_PUNCH"
                self.last_attack_time = now
                return "BACK_PUNCH"
        
        return None
    
    def _is_arm_extended(self, lms, side, tracker) -> bool:
        """Check if the arm is extended (wrist past elbow relative to body center)."""
        if side == "right":
            wrist_idx, elbow_idx = PT.RIGHT_WRIST, PT.RIGHT_ELBOW
        else:
            wrist_idx, elbow_idx = PT.LEFT_WRIST, PT.LEFT_ELBOW
        
        w_rel = tracker.get_body_relative(lms, wrist_idx)
        e_rel = tracker.get_body_relative(lms, elbow_idx)
        
        if w_rel[0] is None or e_rel[0] is None:
            return False
        
        # Wrist should be further from body center than elbow
        wrist_dist = abs(w_rel[0])  # Distance from center
        elbow_dist = abs(e_rel[0])
        
        return wrist_dist > elbow_dist + self.thresh.punch_extension * 0.5
    
    # ─── KICK DETECTION ──────────────────────────────────────────
    
    def _detect_kicks(self, lms, tracker, now) -> Optional[str]:
        """
        Detect kicks based on knee/ankle movement.
        A kick = knee rises from resting position + leg has velocity.
        
        Much more natural than requiring ankle above knee —
        detects forward kicks, side kicks, and knee raises.
        
        Right leg = FRONT_KICK, Left leg = BACK_KICK
        """
        # Right leg → FRONT KICK
        if self._can_fire("FRONT_KICK", now):
            r_speed = self._vel["r_ankle"].get_speed()
            r_kicking = self._is_kicking(lms, "right")
            
            if r_speed > self.thresh.kick_velocity and r_kicking:
                self._set_cooldown("FRONT_KICK", now, self.thresh.kick_cooldown)
                self.last_attack = "FRONT_KICK"
                self.last_attack_time = now
                return "FRONT_KICK"
        
        # Left leg → BACK KICK
        if self._can_fire("BACK_KICK", now):
            l_speed = self._vel["l_ankle"].get_speed()
            l_kicking = self._is_kicking(lms, "left")
            
            if l_speed > self.thresh.kick_velocity and l_kicking:
                self._set_cooldown("BACK_KICK", now, self.thresh.kick_cooldown)
                self.last_attack = "BACK_KICK"
                self.last_attack_time = now
                return "BACK_KICK"
        
        return None
    
    def _is_kicking(self, lms, side) -> bool:
        """
        Check if a leg is in a kicking position.
        
        Uses two checks (either one is sufficient):
        1. Knee has risen: knee Y is significantly above ankle Y 
           compared to the other (standing) leg
        2. Ankle has risen: ankle Y is above its resting position
           (resting = roughly at the same height as the other ankle)
        
        This works for any kick style — forward, upward, side.
        """
        if side == "right":
            hip_idx = PT.RIGHT_HIP
            knee_idx = PT.RIGHT_KNEE
            ankle_idx = PT.RIGHT_ANKLE
            other_ankle_idx = PT.LEFT_ANKLE
            other_knee_idx = PT.LEFT_KNEE
        else:
            hip_idx = PT.LEFT_HIP
            knee_idx = PT.LEFT_KNEE
            ankle_idx = PT.LEFT_ANKLE
            other_ankle_idx = PT.RIGHT_ANKLE
            other_knee_idx = PT.RIGHT_KNEE
        
        hip = lms.get(hip_idx)
        knee = lms.get(knee_idx)
        ankle = lms.get(ankle_idx)
        other_ankle = lms.get(other_ankle_idx)
        other_knee = lms.get(other_knee_idx)
        
        if not all([hip, knee, ankle]):
            return False
        
        # Check 1: Knee has risen — the kicking leg's knee is higher (lower Y)
        #   than the other leg's knee by a threshold
        if other_knee:
            knee_rise = other_knee[1] - knee[1]  # Positive = kicking knee is higher
            if knee_rise > self.thresh.kick_ankle_rise:
                return True
        
        # Check 2: Ankle has risen above the other foot's ankle level
        if other_ankle:
            ankle_rise = other_ankle[1] - ankle[1]  # Positive = ankle is higher
            if ankle_rise > self.thresh.kick_ankle_rise:
                return True
        
        # Check 3: The leg angle (hip-knee-ankle) has decreased significantly
        #   Standing ≈ 170°, small raise ≈ 155°, deep kick ≈ 90-130°
        leg_angle = GestureEngine._compute_knee_angle(hip, knee, ankle)
        if leg_angle < self.thresh.kick_leg_angle:
            return True
        
        return False
    
    # ─── FATAL BLOW DETECTION (U + O) ────────────────────────────
    
    def _detect_fatal_blow(self, lms, now) -> Optional[str]:
        """
        Detect Fatal Blow: both arms spread wide to the sides (T-pose).
        """
        if not self._can_fire("FATAL_BLOW", now):
            return None
        
        rw = lms.get(PT.RIGHT_WRIST)
        lw = lms.get(PT.LEFT_WRIST)
        ls = lms.get(PT.LEFT_SHOULDER)
        rs = lms.get(PT.RIGHT_SHOULDER)
        
        if not all([rw, lw, ls, rs]):
            self._fatal_start = None
            return None
        
        # Wrist-to-wrist horizontal separation
        wrist_sep = abs(rw[0] - lw[0])
        
        # Shoulder width for scaling
        sw = abs(ls[0] - rs[0])
        if sw < 0.01:
            self._fatal_start = None
            return None
        
        # Check 1: Arms spread wide — wrists much wider than shoulders
        spread_ratio = wrist_sep / sw
        arms_wide = spread_ratio > self.thresh.fatal_arms_spread
        
        # Check 2: Wrists at roughly shoulder height
        shoulder_y = (ls[1] + rs[1]) / 2
        rw_y_diff = abs(rw[1] - shoulder_y)
        lw_y_diff = abs(lw[1] - shoulder_y)
        rw_at_height = rw_y_diff < self.thresh.fatal_wrist_height
        lw_at_height = lw_y_diff < self.thresh.fatal_wrist_height
        wrists_at_shoulder = rw_at_height and lw_at_height
        
        

        
        if arms_wide and wrists_at_shoulder:
            if self._fatal_start is None:
                self._fatal_start = now
                print(f"[FATAL] Pose detected! Confirming...")
            
            if now - self._fatal_start > self.thresh.fatal_hold_time:
                self._set_cooldown("FATAL_BLOW", now, self.thresh.fatal_cooldown)
                self.last_attack = "FATAL_BLOW"
                self.last_attack_time = now
                self._fatal_start = None
                print(f">>> FATAL BLOW TRIGGERED!")
                return "FATAL_BLOW"
        else:
            self._fatal_start = None
        
        return None
    
    # ─── BLOCK DETECTION ─────────────────────────────────────────
    
    def _detect_block(self, lms, now) -> bool:
        """
        Detect block: both wrists close to chest / crossed arms.
        Returns True while blocking (hold behavior).
        """
        rw = lms.get(PT.RIGHT_WRIST)
        lw = lms.get(PT.LEFT_WRIST)
        ls = lms.get(PT.LEFT_SHOULDER)
        rs = lms.get(PT.RIGHT_SHOULDER)
        
        if not all([rw, lw, ls, rs]):
            self._block_start = None
            self._is_blocking = False
            return False
        
        # Chest center
        chest_x = (ls[0] + rs[0]) / 2
        chest_y = (ls[1] + rs[1]) / 2
        
        # Distance of each wrist from chest center
        rw_dist = ((rw[0] - chest_x)**2 + (rw[1] - chest_y)**2)**0.5
        lw_dist = ((lw[0] - chest_x)**2 + (lw[1] - chest_y)**2)**0.5
        
        # Both wrists must be close to chest
        block_radius = self.thresh.block_wrist_distance * self._shoulder_width
        wrists_close = rw_dist < block_radius and lw_dist < block_radius
        
        # Additionally: wrists should be crossed (right wrist on left side and vice versa)
        # OR simply both wrists very close together (covering chest)
        wrist_separation = ((rw[0] - lw[0])**2 + (rw[1] - lw[1])**2)**0.5
        wrists_together = wrist_separation < self._shoulder_width * 0.6
        
        if wrists_close and wrists_together:
            if self._block_start is None:
                self._block_start = now
            
            if now - self._block_start > self.thresh.block_hold_time:
                self._is_blocking = True
                return True
        else:
            self._block_start = None
            self._is_blocking = False
        
        return False
    
    # ─── THROW DETECTION ─────────────────────────────────────────
    
    def _detect_throw(self, lms, now) -> bool:
        """Detect throw: both hands thrust forward simultaneously."""
        if not self._can_fire("THROW", now):
            return False
        
        r_speed = self._vel["r_wrist"].get_speed()
        l_speed = self._vel["l_wrist"].get_speed()
        
        # Both wrists must have high velocity
        both_fast = (r_speed > self.thresh.throw_velocity and 
                     l_speed > self.thresh.throw_velocity)
        
        if both_fast:
            # Both wrists should be moving in roughly the same direction
            rvx, rvy = self._vel["r_wrist"].get_velocity()
            lvx, lvy = self._vel["l_wrist"].get_velocity()
            
            # Dot product should be positive (same direction)
            dot = rvx * lvx + rvy * lvy
            if dot > 0:
                self._set_cooldown("THROW", now, self.thresh.throw_cooldown)
                self.last_attack = "THROW"
                self.last_attack_time = now
                return True
        
        return False
    
    # ─── MOVEMENT DETECTION (Spine Midline Angle) ─────────────────
    
    def _detect_movement(self, lms, tracker, now) -> Tuple[Optional[str], float, float]:
        """
        Detect left/right movement using the upper-body midline angle.
        
        Middle axis = line from hip_center to shoulder_center.
        When upright, this line is vertical (90° to X-axis).
        When leaning, the acute angle drops.
        If acute angle < 75° → activate MOVE_LEFT or MOVE_RIGHT
        based on which side the shoulders are offset (+X or -X).
        
        Returns: (action_name, joy_x, joy_y)
        """
        ls = lms.get(PT.LEFT_SHOULDER)
        rs = lms.get(PT.RIGHT_SHOULDER)
        lh = lms.get(PT.LEFT_HIP)
        rh = lms.get(PT.RIGHT_HIP)
        
        if not all([ls, rs, lh, rh]):
            return None, 0.0, 0.0
        
        # Midpoints
        shoulder_cx = (ls[0] + rs[0]) / 2
        shoulder_cy = (ls[1] + rs[1]) / 2
        hip_cx = (lh[0] + rh[0]) / 2
        hip_cy = (lh[1] + rh[1]) / 2
        
        # Middle axis vector: hip_center → shoulder_center
        dx = shoulder_cx - hip_cx   # +X = shoulders shifted right
        dy = shoulder_cy - hip_cy   # -Y = shoulders above hips (screen coords)
        
        # Acute angle between midline and X-axis
        # atan2(|dy|, |dx|) gives the angle from X-axis to the vector
        # When upright: |dy| >> |dx| → angle ≈ 90°
        # When leaning: |dx| grows → angle decreases
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dy < 0.001:  # Prevent division issues if body is horizontal
            acute_angle = 0.0
        else:
            acute_angle = math.degrees(math.atan2(abs_dy, abs_dx))
        
        self.spine_angle = acute_angle  # Store for UI display
        
        # Activation threshold: angle < 75° means significant lean
        MOVE_ANGLE_THRESHOLD = self.thresh.move_angle_threshold
        
        # Compute joy_x from the angle (0° at full lean, 90° at upright)
        # Map the range [THRESHOLD..90] → [±1..0]
        if acute_angle < MOVE_ANGLE_THRESHOLD:
            # Fully activated — clamp to 1.0
            joy_x = 1.0
        elif acute_angle >= 90.0:
            joy_x = 0.0
        else:
            # Gradual ramp: 90° → 0.0, THRESHOLD → 1.0
            joy_x = (90.0 - acute_angle) / (90.0 - MOVE_ANGLE_THRESHOLD)
            joy_x = max(0.0, min(1.0, joy_x))
        
        # Apply direction: dx > 0 means shoulders right of hips → lean right
        if dx < 0:
            joy_x = -joy_x  # Leaning left → negative
        
        # Determine action
        action = None
        if acute_angle < MOVE_ANGLE_THRESHOLD:
            if dx > 0:
                action = "MOVE_RIGHT"
            elif dx < 0:
                action = "MOVE_LEFT"
        
        return action, joy_x, 0.0
    
    # ─── JUMP / DUCK DETECTION (Knee-Angle Based) ─────────────────
    
    @staticmethod
    def _compute_knee_angle(hip, knee, ankle):
        """
        Compute the angle at the knee joint (in degrees).
        Uses the 2D positions of hip, knee, ankle.
        Standing straight ≈ 170°, crouching ≈ 100-130°.
        """
        # Vectors: knee→hip and knee→ankle
        v1x = hip[0] - knee[0]
        v1y = hip[1] - knee[1]
        v2x = ankle[0] - knee[0]
        v2y = ankle[1] - knee[1]
        
        # Dot product and magnitudes
        dot = v1x * v2x + v1y * v2y
        mag1 = (v1x**2 + v1y**2)**0.5
        mag2 = (v2x**2 + v2y**2)**0.5
        
        if mag1 < 0.001 or mag2 < 0.001:
            return 180.0  # Default to straight
        
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))
    
    def _detect_jump_duck(self, lms, tracker, now) -> Optional[str]:
        """
        Detect jump and duck using KNEE ANGLE (camera-distance invariant).
        
        Duck: Knee angle < duck_knee_angle (bending knees = crouching)
        Jump: Hip moving upward + knees straight (not crouching)
        
        Walking toward/away from camera doesn't change knee angle,
        so this prevents false triggers from distance changes.
        """
        # Get required landmarks
        l_hip = lms.get(PT.LEFT_HIP)
        r_hip = lms.get(PT.RIGHT_HIP)
        l_knee = lms.get(PT.LEFT_KNEE)
        r_knee = lms.get(PT.RIGHT_KNEE)
        l_ankle = lms.get(PT.LEFT_ANKLE)
        r_ankle = lms.get(PT.RIGHT_ANKLE)
        
        if not all([l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]):
            return None
        
        # Compute knee angles (average of both legs for stability)
        left_knee_angle = self._compute_knee_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = self._compute_knee_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Update move_y for UI joystick: map knee bend to Y axis
        # 170° = standing (0.0), 120° = deep crouch (-1.0)
        standing_angle = 170.0
        crouch_angle = self.thresh.duck_knee_angle
        if standing_angle > crouch_angle:
            bend_ratio = (standing_angle - avg_knee_angle) / (standing_angle - crouch_angle)
            self.move_y = -max(0.0, min(1.0, bend_ratio))  # Negative = crouch
        
        # ── DUCK: Knees bent below threshold ──
        if avg_knee_angle < self.thresh.duck_knee_angle:
            if self._duck_start is None:
                self._duck_start = now
            
            if now - self._duck_start > self.thresh.duck_hold_time:
                self._is_ducking = True
                return "DUCK"
        else:
            self._duck_start = None
            self._is_ducking = False
        
        # ── JUMP: Hip moving upward + legs straight ──
        if tracker.standing_hip_y is not None:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            delta_y = hip_y - tracker.standing_hip_y  # Negative = moved up
            
            # Knees must be straight (not crouching-and-standing-up)
            knees_straight = avg_knee_angle > self.thresh.jump_knee_angle
            
            # Hip must have risen above standing position
            hip_risen = delta_y < -self.thresh.jump_hip_rise
            
            # Must have upward velocity
            _, hip_vy = self._vel["hip_center"].get_velocity()
            has_upward_velocity = hip_vy < -self.thresh.jump_hip_velocity
            
            if knees_straight and hip_risen and has_upward_velocity:
                if self._can_fire("JUMP", now):
                    self._set_cooldown("JUMP", now, self.thresh.jump_cooldown)
                    # Update move_y for jump flash
                    self.move_y = 1.0
                    return "JUMP"
        
        return None
    
    # ─── FLIP STANCE ─────────────────────────────────────────────
    
    def _detect_flip_stance(self, lms, now) -> Optional[str]:
        """Detect quick body twist (shoulder rotation)."""
        if not self._can_fire("FLIP_STANCE", now):
            return None
        
        ls = lms.get(PT.LEFT_SHOULDER)
        rs = lms.get(PT.RIGHT_SHOULDER)
        
        if not ls or not rs:
            return None
        
        # Use Z-depth difference between shoulders as rotation indicator
        z_diff = abs(ls[2] - rs[2])
        
        # If shoulders are at significantly different Z depths, body is twisted
        if z_diff > self.thresh.flip_rotation_speed * self._shoulder_width:
            self._set_cooldown("FLIP_STANCE", now, self.thresh.flip_cooldown)
            return "FLIP_STANCE"
        
        return None
    
    # ─── CHARACTER ASSIST ────────────────────────────────────────
    
    def _detect_char_assist(self, lms, now) -> Optional[str]:
        """Detect both hands raised above head."""
        if not self._can_fire("CHAR_ASSIST", now):
            return None
        
        rw = lms.get(PT.RIGHT_WRIST)
        lw = lms.get(PT.LEFT_WRIST)
        nose = lms.get(PT.NOSE)
        
        if not all([rw, lw, nose]):
            self._assist_start = None
            return None
        
        # Both wrists must be above nose
        both_above = rw[1] < nose[1] - self.thresh.assist_wrist_above_head and \
                     lw[1] < nose[1] - self.thresh.assist_wrist_above_head
        
        if both_above:
            if self._assist_start is None:
                self._assist_start = now
            
            if now - self._assist_start > self.thresh.assist_hold_time:
                self._set_cooldown("CHAR_ASSIST", now, self.thresh.assist_cooldown)
                self._assist_start = None
                return "CHAR_ASSIST"
        else:
            self._assist_start = None
        
        return None
    
    # ─── COMBO DETECTION ─────────────────────────────────────────
    
    def _detect_combo(self, now) -> Optional[str]:
        """Check if recent actions match any combo sequence."""
        if now - self._last_combo_time < 0.5:  # Global combo cooldown
            return None
        
        for combo in self.combos:
            if self._check_combo_match(combo, now):
                self._last_combo_time = now
                # Clear the action log to prevent double-triggers
                self._action_log.clear()
                return combo.name
        
        return None
    
    def _check_combo_match(self, combo: ComboDefinition, now: float) -> bool:
        """Check if the action log matches a combo's trigger sequence."""
        seq = combo.trigger_sequence
        if len(self._action_log) < len(seq):
            return False
        
        # Look for the sequence in recent actions within the time window
        recent = [(a, t) for a, t in self._action_log if now - t < combo.time_window]
        
        if len(recent) < len(seq):
            return False
        
        # Check if the last N actions match the sequence
        last_n = recent[-len(seq):]
        for i, (action, _) in enumerate(last_n):
            if action != seq[i]:
                return False
        
        return True
    
    # ─── START/STOP GESTURES ─────────────────────────────────────
    
    def check_gestures(self, landmarks) -> Optional[str]:
        """
        Check for start/stop gestures using pose landmarks.
        START: Both hands raised high above head
        STOP: Both hands on hips (wrists near hips)
        """
        if not landmarks:
            return None
        
        rw = landmarks.get(PT.RIGHT_WRIST)
        lw = landmarks.get(PT.LEFT_WRIST)
        nose = landmarks.get(PT.NOSE)
        rh = landmarks.get(PT.RIGHT_HIP)
        lh = landmarks.get(PT.LEFT_HIP)
        
        if not all([rw, lw]):
            return None
        
        # START: Both wrists well above head
        if nose:
            both_way_above = (rw[1] < nose[1] - 0.15 and lw[1] < nose[1] - 0.15)
            if both_way_above:
                return "START"
        
        # STOP: Both wrists near hips (hands on hips gesture)
        if rh and lh:
            r_near_hip = ((rw[0] - rh[0])**2 + (rw[1] - rh[1])**2)**0.5 < 0.08
            l_near_hip = ((lw[0] - lh[0])**2 + (lw[1] - lh[1])**2)**0.5 < 0.08
            if r_near_hip and l_near_hip:
                return "STOP"
        
        return None
