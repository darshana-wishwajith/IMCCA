"""
swipe_detector.py - Swipe-Based Motion Detection for Fighting Games

Hand gesture gating:
- OPEN hand (5 fingers) = swipe detection active, direction held
- FIST (closed hand) = neutral / release all directions
- Fist→Open transition = set new neutral at current palm position

Swipe mechanics:
- Swipe left/right/up/down → digital D-Pad press (-1, 0, or +1)
- Per-axis tracking: X and Y activate/deactivate independently
- Velocity spike = fast trigger; displacement = backup trigger
- Diagonals supported (both axes active simultaneously)

Designed for Mortal Kombat 11 where D-Pad digital inputs are
superior to analog joystick proportional control.
"""

import time
import numpy as np
from collections import deque


class SwipeMotionDetector:
    def __init__(self):
        # --- POSE SMOOTHING (same as MotionDetector) ---
        self.ema_alpha = 0.8
        self.prev_pose_lms = None
        self.last_frame_t = None
        self.is_calibrated = True
        self.is_fist = False  # Tracks current fist state

        # --- SWIPE THRESHOLDS ---
        self.SWIPE_VELOCITY_THRESH = 0.6    # norm coords/sec to trigger via speed
        self.ACTIVATION_DISPLACEMENT = 0.03 # norm coords displacement to trigger
        self.NEUTRAL_ZONE_RADIUS = 0.02     # return within this to release (< ACTIVATION)

        # --- PALM TRACKING ---
        self._palm_history = deque(maxlen=5)
        self._neutral_x = None
        self._neutral_y = None
        self._smooth_palm_x = None
        self._smooth_palm_y = None
        self._palm_ema = 0.7  # Higher = less smoothing = faster response

        # --- PER-AXIS SWIPE STATE ---
        self._x_active = False
        self._y_active = False
        self._dir_x = 0   # -1 (left), 0, +1 (right)
        self._dir_y = 0   # -1 (down), 0, +1 (up)

        # --- HAND OPEN/FIST STATE ---
        self._hand_is_open = False  # True when all 5 fingers extended
        self._finger_count = 0      # Current finger count for UI
        self._fist_frame_count = 0  # Consecutive frames fist detected (temporal buffer)
        self.FIST_CONFIRM_FRAMES = 3  # Need 3 consecutive fist frames to confirm

        # --- HAND TRACKING ---
        self._last_hand_seen = 0
        self._hand_timeout = 0.5

        # --- RIGHT HAND ATTACK STATE ---
        self._rh_prev_x = None              # Previous frame raw palm X
        self._rh_prev_y = None              # Previous frame raw palm Y
        self._rh_last_seen = 0
        self._rh_finger_count = 0
        self._rh_cooldown_until = 0         # Ignore attacks until this time
        self._rh_blocking = False           # Currently holding block
        self._rh_fist_still_since = None    # When fist became still
        self._last_attack = None            # Last attack name for UI
        self._last_attack_time = 0          # When last attack fired

        # --- ATTACK THRESHOLDS ---
        # Frame-to-frame displacement to trigger (NOT velocity)
        # At 30fps: 0.015 normalized ≈ light flick, 0.03 ≈ medium motion
        self.ATTACK_DISP_THRESH = 0.015     # Per-frame displacement to trigger (adjustable with [ ] keys)
        self.ATTACK_COOLDOWN = 0.10         # Seconds between attacks (100ms)
        self.BLOCK_DISP_MAX = 0.005         # Max per-frame displacement for "still"
        self.BLOCK_HOLD_TIME = 0.15         # Seconds of still fist to start block

        # --- UI COMPATIBILITY ---
        self.JOY_MAX_REACH = 0.06  # kept for draw_ui compatibility

    @property
    def swipe_state(self):
        """Current state string for UI display."""
        hand_lbl = "OPEN" if self._hand_is_open else "FIST"
        if self._x_active or self._y_active:
            dirs = []
            if self._dir_y > 0: dirs.append("UP")
            if self._dir_y < 0: dirs.append("DN")
            if self._dir_x < 0: dirs.append("L")
            if self._dir_x > 0: dirs.append("R")
            return f"{hand_lbl} HOLD " + "-".join(dirs)
        return f"{hand_lbl} IDLE"

    @property
    def rh_state(self):
        """Right hand state string for UI."""
        if self._rh_blocking:
            return "BLOCK"
        if self._last_attack and (time.time() - self._last_attack_time < 0.3):
            return self._last_attack
        return f"RH:{self._rh_finger_count}"

    # ── finger counting (distance-based, rotation-invariant) ──────

    @staticmethod
    def _dist(a, b):
        """2D Euclidean distance between two landmark tuples."""
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _count_fingers(self, hand_lms):
        """Count extended fingers using distance-from-wrist method.
        
        A finger is 'extended' if its TIP is farther from the WRIST
        than its PIP/IP joint. This works regardless of hand rotation
        (up, down, sideways) unlike Y-position checks.
        """
        lms = hand_lms
        wrist = lms[0]

        # Thumb: tip(4) vs IP(3) distance from wrist
        thumb = self._dist(lms[4], wrist) > self._dist(lms[3], wrist)
        # Index: tip(8) vs PIP(6)
        index = self._dist(lms[8], wrist) > self._dist(lms[6], wrist)
        # Middle: tip(12) vs PIP(10)
        middle = self._dist(lms[12], wrist) > self._dist(lms[10], wrist)
        # Ring: tip(16) vs PIP(14)
        ring = self._dist(lms[16], wrist) > self._dist(lms[14], wrist)
        # Pinky: tip(20) vs PIP(18)
        pinky = self._dist(lms[20], wrist) > self._dist(lms[18], wrist)

        return sum([thumb, index, middle, ring, pinky])

    # ── helpers (identical to MotionDetector) ──────────────────────

    def _get_palm_center(self, hand_lms):
        keys = [0, 5, 9, 13, 17]
        xs = [hand_lms[k][0] for k in keys if k in hand_lms]
        ys = [hand_lms[k][1] for k in keys if k in hand_lms]
        if not xs: return None, None
        return sum(xs)/len(xs), sum(ys)/len(ys)

    def _find_left_hand(self, hands_lms, pose_lms):
        if not hands_lms: return None
        if pose_lms and 16 in pose_lms:
            px, py = pose_lms[16][0], pose_lms[16][1]
            best, best_d = None, float('inf')
            for h in hands_lms:
                if h["label"] == "Left":
                    hx, hy = self._get_palm_center(h["landmarks"])
                    if hx is not None:
                        d = ((hx-px)**2 + (hy-py)**2)**0.5
                        if d < best_d and d < 0.2:
                            best_d, best = d, h
            if best: return best
        for h in hands_lms:
            if h["label"] == "Left": return h
        return None

    def _find_right_hand(self, hands_lms, pose_lms):
        """Find user's physical RIGHT hand."""
        if not hands_lms: return None
        if pose_lms and 15 in pose_lms:
            px, py = pose_lms[15][0], pose_lms[15][1]
            best, best_d = None, float('inf')
            for h in hands_lms:
                if h["label"] == "Right":
                    hx, hy = self._get_palm_center(h["landmarks"])
                    if hx is not None:
                        d = ((hx-px)**2 + (hy-py)**2)**0.5
                        if d < best_d and d < 0.2:
                            best_d, best = d, h
            if best: return best
        for h in hands_lms:
            if h["label"] == "Right": return h
        return None

    def _compute_velocity(self):
        if len(self._palm_history) < 3:
            return 0.0, 0.0
        old, new = self._palm_history[0], self._palm_history[-1]
        dt = new[2] - old[2]
        if dt < 0.001: return 0.0, 0.0
        return (new[0]-old[0])/dt, (new[1]-old[1])/dt

    def _smooth_pose(self, curr, prev):
        if prev is None: return curr
        out = {}
        for idx in curr:
            if idx in prev:
                c = np.array(curr[idx][:3]); p = np.array(prev[idx][:3])
                s = self.ema_alpha*c + (1-self.ema_alpha)*p
                out[idx] = (*s, curr[idx][3])
            else:
                out[idx] = curr[idx]
        return out

    # ── main detection ─────────────────────────────────────────────

    def detect(self, pose_lms, hands_lms=None):
        if not pose_lms:
            return [], 0.0, 0.0

        now = time.time()
        self.last_frame_t = now
        smoothed = self._smooth_pose(pose_lms, self.prev_pose_lms)

        left_hand = self._find_left_hand(hands_lms, pose_lms)

        if left_hand:
            hlms = left_hand["landmarks"]
            palm_x, palm_y = self._get_palm_center(hlms)

            # ── FINGER GATING: fist vs open hand ──
            fingers = self._count_fingers(hlms)
            self._finger_count = fingers

            if fingers <= 1:
                # Increment fist frame counter (temporal buffer)
                self._fist_frame_count += 1
            else:
                self._fist_frame_count = 0  # Reset if not fist

            # Only confirm fist after N consecutive fist frames
            fist_confirmed = self._fist_frame_count >= self.FIST_CONFIRM_FRAMES

            if fist_confirmed:
                # ── FIST CONFIRMED: release everything, go to neutral ──
                self.is_fist = True
                if self._hand_is_open:  # Transition: open → fist
                    self._hand_is_open = False
                    self._x_active = self._y_active = False
                    self._dir_x = self._dir_y = 0
                    self._palm_history.clear()
                # Still track palm smoothing so transition back is smooth
                if palm_x is not None:
                    if self._smooth_palm_x is None:
                        self._smooth_palm_x, self._smooth_palm_y = palm_x, palm_y
                    else:
                        a = self._palm_ema
                        self._smooth_palm_x = a*palm_x + (1-a)*self._smooth_palm_x
                        self._smooth_palm_y = a*palm_y + (1-a)*self._smooth_palm_y
                    self._last_hand_seen = now

            elif fingers >= 4:
                # ── OPEN HAND: swipe detection active ──
                self.is_fist = False

                if palm_x is not None:
                    # Light EMA smoothing
                    if self._smooth_palm_x is None:
                        self._smooth_palm_x, self._smooth_palm_y = palm_x, palm_y
                    else:
                        a = self._palm_ema
                        self._smooth_palm_x = a*palm_x + (1-a)*self._smooth_palm_x
                        self._smooth_palm_y = a*palm_y + (1-a)*self._smooth_palm_y

                    px, py = self._smooth_palm_x, self._smooth_palm_y
                    self._palm_history.append((px, py, now))

                    # Fist→Open transition OR first detection: set new neutral
                    if (not self._hand_is_open or
                            self._neutral_x is None or
                            (now - self._last_hand_seen > self._hand_timeout)):
                        self._neutral_x, self._neutral_y = px, py
                        self._x_active = self._y_active = False
                        self._dir_x = self._dir_y = 0
                        self._palm_history.clear()
                        self._palm_history.append((px, py, now))
                        self._hand_is_open = True
                        print(f">>> [SWIPE] OPEN — Neutral set at ({px:.3f}, {py:.3f})")

                    self._last_hand_seen = now

                    # Displacement from neutral
                    dx = px - self._neutral_x   # + = right
                    dy = py - self._neutral_y   # + = down on screen

                    # Velocity
                    vel_x, vel_y = self._compute_velocity()

                    # ── X AXIS ──
                    if not self._x_active:
                        if (abs(dx) > self.ACTIVATION_DISPLACEMENT or
                                (abs(vel_x) > self.SWIPE_VELOCITY_THRESH and
                                 abs(dx) > self.NEUTRAL_ZONE_RADIUS)):
                            self._x_active = True
                            self._dir_x = 1 if dx > 0 else -1
                    else:
                        if abs(dx) < self.NEUTRAL_ZONE_RADIUS:
                            self._x_active = False
                            self._dir_x = 0
                        else:
                            self._dir_x = 1 if dx > 0 else -1

                    # ── Y AXIS ──  (inverted: screen-down → joy-down = -1)
                    if not self._y_active:
                        if (abs(dy) > self.ACTIVATION_DISPLACEMENT or
                                (abs(vel_y) > self.SWIPE_VELOCITY_THRESH and
                                 abs(dy) > self.NEUTRAL_ZONE_RADIUS)):
                            self._y_active = True
                            self._dir_y = -1 if dy > 0 else 1
                    else:
                        if abs(dy) < self.NEUTRAL_ZONE_RADIUS:
                            self._y_active = False
                            self._dir_y = 0
                        else:
                            self._dir_y = -1 if dy > 0 else 1

            # else: 2-3 fingers = intermediate, keep current state (prevents flicker)

        else:
            # Hand lost → release everything
            if self._x_active or self._y_active:
                self._x_active = self._y_active = False
                self._dir_x = self._dir_y = 0
            self._hand_is_open = False

        # ── RIGHT HAND ATTACKS ─────────────────────────────────────
        actions = []
        right_hand = self._find_right_hand(hands_lms, pose_lms)

        if right_hand:
            rh_lms = right_hand["landmarks"]
            rh_px, rh_py = self._get_palm_center(rh_lms)

            if rh_px is not None:
                self._rh_last_seen = now

                # Finger count (distance-based)
                rh_fingers = self._count_fingers(rh_lms)
                self._rh_finger_count = rh_fingers
                rh_is_fist = rh_fingers <= 1
                rh_is_open = rh_fingers >= 4

                # Frame-to-frame displacement (NO smoothing = instant)
                frame_disp = 0.0
                if self._rh_prev_x is not None:
                    dx = rh_px - self._rh_prev_x
                    dy = rh_py - self._rh_prev_y
                    frame_disp = (dx**2 + dy**2)**0.5

                # ── PUNCH / KICK (instant displacement trigger) ──
                if (now > self._rh_cooldown_until
                        and frame_disp > self.ATTACK_DISP_THRESH):
                    if rh_is_fist:
                        actions.append("PUNCH_RIGHT")
                        self._last_attack = "PUNCH"
                        self._last_attack_time = now
                        self._rh_cooldown_until = now + self.ATTACK_COOLDOWN
                        self._rh_blocking = False
                        self._rh_fist_still_since = None
                    elif rh_is_open:
                        actions.append("KICK_RIGHT")
                        self._last_attack = "KICK"
                        self._last_attack_time = now
                        self._rh_cooldown_until = now + self.ATTACK_COOLDOWN
                        self._rh_blocking = False

                # ── BLOCK (fist held still) ──
                if rh_is_fist and frame_disp < self.BLOCK_DISP_MAX:
                    if self._rh_fist_still_since is None:
                        self._rh_fist_still_since = now
                    elif (now - self._rh_fist_still_since > self.BLOCK_HOLD_TIME
                          and now > self._rh_cooldown_until):
                        if not self._rh_blocking:
                            self._rh_blocking = True
                        actions.append("BLOCK")
                else:
                    if self._rh_blocking:
                        self._rh_blocking = False
                    self._rh_fist_still_since = None

                # Store raw position for next frame
                self._rh_prev_x = rh_px
                self._rh_prev_y = rh_py
        else:
            # Right hand lost
            if self._rh_blocking:
                self._rh_blocking = False
            self._rh_fist_still_since = None
            self._rh_prev_x = self._rh_prev_y = None

        self.prev_pose_lms = smoothed
        return actions, float(self._dir_x), float(self._dir_y)

    # ── gestures (identical to MotionDetector) ─────────────────────

    def check_gestures(self, h_list):
        total_up, v_sign = 0, False
        for hand in h_list:
            lms = hand["landmarks"]
            ups = [lms[4][1]<lms[3][1], lms[8][1]<lms[6][1],
                   lms[12][1]<lms[10][1], lms[16][1]<lms[14][1],
                   lms[20][1]<lms[18][1]]
            cnt = sum(ups); total_up += cnt
            if cnt == 2 and ups[1] and ups[2]: v_sign = True
        if total_up == 10: return "START"
        if v_sign: return "STOP"
        return None

    # ── calibration ────────────────────────────────────────────────

    def calibrate(self, pose_lms, hands_lms=None):
        if pose_lms and 11 in pose_lms and 12 in pose_lms:
            ls, rs = pose_lms[11], pose_lms[12]
            sw = ((ls[0]-rs[0])**2 + (ls[1]-rs[1])**2)**0.5
            self.NEUTRAL_ZONE_RADIUS    = max(0.012, sw * 0.08)
            self.ACTIVATION_DISPLACEMENT = max(0.015, sw * 0.10)
            self.SWIPE_VELOCITY_THRESH  = max(0.3, sw * 1.8)
            self.JOY_MAX_REACH = max(0.02, sw * 0.25)
            self.ATTACK_DISP_THRESH = max(0.008, sw * 0.05)
            print(f">>> [SWIPE] Calibrated! Shoulder:{sw:.3f} "
                  f"Neutral:{self.NEUTRAL_ZONE_RADIUS:.3f} "
                  f"Activ:{self.ACTIVATION_DISPLACEMENT:.3f} "
                  f"AtkDisp:{self.ATTACK_DISP_THRESH:.3f}")

        self._neutral_x = self._neutral_y = None
        self._smooth_palm_x = self._smooth_palm_y = None
        self._palm_history.clear()
        self._x_active = self._y_active = False
        self._dir_x = self._dir_y = 0
        self._hand_is_open = False
        self._finger_count = 0
        self._fist_frame_count = 0
        self.is_fist = False
        # Right hand reset
        self._rh_prev_x = self._rh_prev_y = None
        self._rh_cooldown_until = 0
        self._rh_blocking = False
        self._rh_fist_still_since = None
        self._last_attack = None
        self.is_calibrated = True
        return True

    def reset_neutral(self):
        """Full neutral reset (called by 'R' key)."""
        self._neutral_x = self._neutral_y = None
        self._smooth_palm_x = self._smooth_palm_y = None
        self._palm_history.clear()
        self._x_active = self._y_active = False
        self._dir_x = self._dir_y = 0
        self._hand_is_open = False
        self._finger_count = 0
        self._fist_frame_count = 0
        self.is_fist = False
        # Right hand reset
        self._rh_prev_x = self._rh_prev_y = None
        self._rh_cooldown_until = 0
        self._rh_blocking = False
        self._rh_fist_still_since = None
        self._last_attack = None
        print(">>> [SWIPE] Full RESET")
