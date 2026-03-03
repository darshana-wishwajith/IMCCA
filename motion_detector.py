"""
motion_detector.py (MODULE 2)
RELATIVE HAND TRACKING - Joystick via palm position

Purpose:
- Tracks the user's LEFT hand (MediaPipe label "Right" in mirrored video)
- Uses palm center position for joystick control
- When hand first appears, its position becomes the neutral center
- Small movements from center = joystick input
- No arm/shoulder tracking needed — works at any body position
"""

import time
import numpy as np

class MotionDetector:
    def __init__(self):
        # Smoothing
        self.ema_alpha = 0.8
        
        # Internal State
        self.prev_pose_lms = None
        self.last_frame_t = None
        self.is_calibrated = True
        self.is_fist = False  # Keep for UI compatibility
        
        # --- JOYSTICK PARAMETERS ---
        self.JOY_DEADZONE = 0.08     # Ignore tiny jitter
        self.JOY_MAX_REACH = 0.06    # Max hand displacement for full joystick (in normalized coords)
                                      # ~0.06 = very small hand movements = full range
        self.JOY_SNAP_RATIO = 1.8    # Snap-to-axis ratio
        
        # --- RELATIVE TRACKING STATE ---
        self._neutral_x = None       # Neutral palm center X
        self._neutral_y = None       # Neutral palm center Y
        self._last_hand_seen = 0     # Timestamp of last hand detection
        self._hand_timeout = 0.5     # Reset neutral after hand gone for 0.5s
        self._smooth_palm_x = None   # EMA-smoothed palm position
        self._smooth_palm_y = None
        self._palm_ema = 0.5         # Palm smoothing factor (0.5 = balanced)

    def _get_palm_center(self, hand_lms):
        """Average of wrist + 4 MCP joints = stable palm center."""
        keys = [0, 5, 9, 13, 17]
        xs = [hand_lms[k][0] for k in keys if k in hand_lms]
        ys = [hand_lms[k][1] for k in keys if k in hand_lms]
        if not xs: return None, None
        return sum(xs)/len(xs), sum(ys)/len(ys)

    def _find_left_hand(self, hands_lms, pose_lms):
        """Find user's physical LEFT hand.
        Matches hand to the primary player's body (Pose 16) to ignore background people."""
        if not hands_lms: return None
        
        # Use primary player's wrist from Pose (16 = physical left wrist in mirrored)
        if pose_lms and 16 in pose_lms:
            px, py = pose_lms[16][0], pose_lms[16][1]
            best_hand = None
            min_dist = float('inf')
            
            for hand in hands_lms:
                if hand["label"] == "Left":
                    hx, hy = self._get_palm_center(hand["landmarks"])
                    if hx is not None:
                        dist = ((hx - px)**2 + (hy - py)**2)**0.5
                        # Hand must be relatively close to the primary player's wrist (ignore background)
                        if dist < min_dist and dist < 0.2: 
                            min_dist = dist
                            best_hand = hand
            if best_hand: return best_hand
            
        # Fallback if pose matching fails
        for hand in hands_lms:
            if hand["label"] == "Left":
                return hand
        return None

    def _smooth_pose(self, curr, prev):
        if prev is None: return curr
        smoothed = {}
        for idx in curr:
            if idx in prev:
                c_v = np.array(curr[idx][:3])
                p_v = np.array(prev[idx][:3])
                s_v = self.ema_alpha * c_v + (1 - self.ema_alpha) * p_v
                smoothed[idx] = (*s_v, curr[idx][3])
            else:
                smoothed[idx] = curr[idx]
        return smoothed

    def detect(self, pose_lms, hands_lms=None):
        if not pose_lms:
            return [], 0.0, 0.0
        
        now = time.time()
        self.last_frame_t = now
        
        smoothed = self._smooth_pose(pose_lms, self.prev_pose_lms)
        joy_x, joy_y = 0.0, 0.0

        # --- FIND LEFT HAND AND COMPUTE JOYSTICK ---
        left_hand = self._find_left_hand(hands_lms, pose_lms)
        
        if left_hand:
            hlms = left_hand["landmarks"]
            palm_x, palm_y = self._get_palm_center(hlms)
            
            if palm_x is not None:
                # Smooth the palm position
                if self._smooth_palm_x is None:
                    self._smooth_palm_x = palm_x
                    self._smooth_palm_y = palm_y
                else:
                    self._smooth_palm_x = self._palm_ema * palm_x + (1-self._palm_ema) * self._smooth_palm_x
                    self._smooth_palm_y = self._palm_ema * palm_y + (1-self._palm_ema) * self._smooth_palm_y
                
                px, py = self._smooth_palm_x, self._smooth_palm_y
                
                # If no neutral point yet, or hand was gone and came back → set new neutral
                if self._neutral_x is None or (now - self._last_hand_seen > self._hand_timeout):
                    self._neutral_x = px
                    self._neutral_y = py
                    print(f">>> Joystick center set at ({px:.3f}, {py:.3f})")
                
                self._last_hand_seen = now
                
                # Compute displacement from neutral
                dx = px - self._neutral_x   # Positive = right on screen
                dy = py - self._neutral_y   # Positive = down on screen
                
                # Scale to [-1, 1] joystick range
                raw_x = np.clip(dx / self.JOY_MAX_REACH, -1.0, 1.0)
                raw_y = np.clip(-dy / self.JOY_MAX_REACH, -1.0, 1.0)  # Invert Y (up = positive)
                
                # Deadzone
                if abs(raw_x) < self.JOY_DEADZONE: raw_x = 0.0
                if abs(raw_y) < self.JOY_DEADZONE: raw_y = 0.0
                
                # Snap to axis if strongly moving in one cardinal direction
                # (e.g. mostly horizontal -> lock to exact left/right)
                ax, ay = abs(raw_x), abs(raw_y)
                if ax > 0.01 or ay > 0.01:
                    if ay > ax * 1.8:  # Strongly vertical (snap to Up/Down)
                        raw_x = 0.0
                    elif ax > ay * 1.8:  # Strongly horizontal (snap to Left/Right)
                        raw_y = 0.0
                
                joy_x = raw_x
                joy_y = raw_y
        else:
            # Hand not detected → joystick returns to center
            # (Neutral will be reset when hand reappears after timeout)
            pass

        self.prev_pose_lms = smoothed
        return [], joy_x, joy_y

    def check_gestures(self, h_list):
        total_up = 0
        v_sign_detected = False
        for hand in h_list:
            lms = hand["landmarks"]
            thumb_up = lms[4][1] < lms[3][1]
            idx_up = lms[8][1] < lms[6][1]
            mid_up = lms[12][1] < lms[10][1]
            rng_up = lms[16][1] < lms[14][1]
            pnk_up = lms[20][1] < lms[18][1]
            hand_up_count = sum([thumb_up, idx_up, mid_up, rng_up, pnk_up])
            total_up += hand_up_count
            if hand_up_count == 2 and idx_up and mid_up: v_sign_detected = True
        
        if total_up == 10: return "START"
        if v_sign_detected: return "STOP"
        return None

    def calibrate(self, pose_lms, hands_lms=None):
        """Auto-adjust sensitivity based on the user's distance from the camera."""
        if pose_lms and 11 in pose_lms and 12 in pose_lms:
            # 11 = Left shoulder, 12 = Right shoulder (MediaPipe body coordinates)
            ls = pose_lms[11]
            rs = pose_lms[12]
            # Euclidean distance between shoulders on screen
            shoulder_width = ((ls[0] - rs[0])**2 + (ls[1] - rs[1])**2)**0.5
            
            # Max reach becomes proportional to body size on screen.
            # E.g. 0.25 means "moving your hand 25% of your shoulder width = full joystick"
            self.JOY_MAX_REACH = max(0.02, shoulder_width * 0.25)
            print(f">>> Auto-Calibrated Focus! Shoulder Width: {shoulder_width:.3f} | Sensitivity: {self.JOY_MAX_REACH:.3f}")
        
        # Reset neutral center intentionally
        self._neutral_x = None
        self._neutral_y = None
        self._smooth_palm_x = None
        self._smooth_palm_y = None
        self.is_calibrated = True
        return True
