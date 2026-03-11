"""
pose_tracker.py — Optimized Pose Tracking with Threaded Camera
IMCCA v2

Features:
- Threaded camera read → zero frame drops, always latest frame
- MediaPipe Pose ONLY (no Hands model) → maximum FPS
- Landmark smoothing with EMA
- Body-relative coordinate system
- Skeleton drawing
"""

import time
import threading
import cv2
import mediapipe as mp
import numpy as np
from config import CameraConfig, PoseConfig, SmoothingConfig


class ThreadedCamera:
    """Reads camera frames in a background thread for zero-latency capture."""
    
    def __init__(self, cfg: CameraConfig):
        self.cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Could not open webcam.")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        
        self.mirror = cfg.mirror
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        
        # Start reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
        # Wait for first frame
        for _ in range(30):
            with self.lock:
                if self.frame is not None:
                    break
            time.sleep(0.05)
    
    def _reader(self):
        """Continuously reads frames in the background."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
                    self.ret = ret
    
    def read(self):
        """Returns the latest frame (never blocks)."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class PoseTracker:
    """
    Tracks body pose using MediaPipe Pose (BlazePose).
    Provides smoothed landmarks and body-relative coordinate system.
    """
    
    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Skeleton drawing connections (no face landmarks)
    BODY_CONNECTIONS = [
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15),
        # Right arm
        (12, 14), (14, 16),
        # Left leg
        (23, 25), (25, 27),
        # Right leg
        (24, 26), (26, 28),
    ]
    
    CONNECTION_COLORS = {
        (11, 12): (0, 255, 100),   # Torso - green
        (11, 23): (0, 255, 100),
        (12, 24): (0, 255, 100),
        (23, 24): (0, 255, 100),
        (11, 13): (255, 100, 0),   # Left arm - blue
        (13, 15): (255, 100, 0),
        (12, 14): (0, 100, 255),   # Right arm - red
        (14, 16): (0, 100, 255),
        (23, 25): (255, 255, 0),   # Left leg - cyan
        (25, 27): (255, 255, 0),
        (24, 26): (255, 0, 255),   # Right leg - magenta
        (26, 28): (255, 0, 255),
    }
    
    def __init__(self, cam_cfg=None, pose_cfg=None, smooth_cfg=None):
        self.cam_cfg = cam_cfg or CameraConfig()
        self.pose_cfg = pose_cfg or PoseConfig()
        self.smooth_cfg = smooth_cfg or SmoothingConfig()
        
        # Initialize camera
        self.camera = ThreadedCamera(self.cam_cfg)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.pose_cfg.model_complexity,
            smooth_landmarks=self.pose_cfg.smooth_landmarks,
            min_detection_confidence=self.pose_cfg.min_detection_confidence,
            min_tracking_confidence=self.pose_cfg.min_tracking_confidence,
            enable_segmentation=self.pose_cfg.enable_segmentation,
        )
        
        # State
        self._prev_landmarks = None
        self._prev_time = time.time()
        
        # Body reference measurements (set during calibration)
        self.shoulder_width = None    # Normalized shoulder width
        self.torso_height = None      # Normalized torso height (shoulder to hip)
        self.standing_hip_y = None    # Calibrated standing hip height
        self.body_center_x = None     # Calibrated body center X
    
    def read_frame(self):
        """Get the latest camera frame."""
        return self.camera.read()
    
    def process(self, frame):
        """
        Run pose inference and return smoothed landmarks dict.
        Returns: (landmarks_dict, raw_results) where landmarks_dict maps
                 index -> (x, y, z, visibility)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return {}, results
        
        # Extract landmarks
        raw = {}
        for i, lm in enumerate(results.pose_landmarks.landmark):
            raw[i] = (lm.x, lm.y, lm.z, lm.visibility)
        
        # Apply EMA smoothing
        smoothed = self._smooth(raw)
        self._prev_landmarks = smoothed
        
        return smoothed, results
    
    def _smooth(self, current):
        """Apply EMA smoothing to landmarks."""
        if self._prev_landmarks is None:
            return current
        
        alpha = self.smooth_cfg.ema_alpha
        smoothed = {}
        for idx in current:
            if idx in self._prev_landmarks:
                c = np.array(current[idx][:3])
                p = np.array(self._prev_landmarks[idx][:3])
                s = alpha * c + (1 - alpha) * p
                smoothed[idx] = (float(s[0]), float(s[1]), float(s[2]), current[idx][3])
            else:
                smoothed[idx] = current[idx]
        return smoothed
    
    def calibrate(self, landmarks):
        """
        Calibrate body reference measurements from current pose.
        Call this when the user is standing in a neutral position.
        """
        if not landmarks:
            return False
        
        ls = landmarks.get(self.LEFT_SHOULDER)
        rs = landmarks.get(self.RIGHT_SHOULDER)
        lh = landmarks.get(self.LEFT_HIP)
        rh = landmarks.get(self.RIGHT_HIP)
        
        if not all([ls, rs, lh, rh]):
            return False
        
        # Shoulder width (primary scaling reference)
        self.shoulder_width = ((ls[0] - rs[0])**2 + (ls[1] - rs[1])**2)**0.5
        
        # Torso height (shoulder midpoint to hip midpoint)
        shoulder_mid_y = (ls[1] + rs[1]) / 2
        hip_mid_y = (lh[1] + rh[1]) / 2
        self.torso_height = abs(hip_mid_y - shoulder_mid_y)
        
        # Standing hip center
        self.standing_hip_y = hip_mid_y
        self.body_center_x = (ls[0] + rs[0]) / 2
        
        print(f">>> CALIBRATED: shoulders={self.shoulder_width:.3f} "
              f"torso={self.torso_height:.3f} "
              f"hip_y={self.standing_hip_y:.3f} "
              f"center_x={self.body_center_x:.3f}")
        return True
    
    def get_body_relative(self, landmarks, point_idx):
        """
        Convert a landmark position to body-relative coordinates.
        Returns (rel_x, rel_y) where units are in 'shoulder widths'.
        (0,0) = body center (mid-shoulders), positive X = right, positive Y = down.
        """
        if not landmarks or point_idx not in landmarks:
            return None, None
        
        if self.shoulder_width is None or self.shoulder_width < 0.01:
            return None, None
        
        point = landmarks[point_idx]
        ls = landmarks.get(self.LEFT_SHOULDER)
        rs = landmarks.get(self.RIGHT_SHOULDER)
        
        if not ls or not rs:
            return None, None
        
        center_x = (ls[0] + rs[0]) / 2
        center_y = (ls[1] + rs[1]) / 2
        
        rel_x = (point[0] - center_x) / self.shoulder_width
        rel_y = (point[1] - center_y) / self.shoulder_width
        
        return rel_x, rel_y
    
    def draw_skeleton(self, frame, results):
        """Draw the body skeleton on the frame (no face)."""
        if not results.pose_landmarks:
            return
        
        lms = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        # Draw connections
        for (a, b) in self.BODY_CONNECTIONS:
            if lms[a].visibility > 0.5 and lms[b].visibility > 0.5:
                pt1 = (int(lms[a].x * w), int(lms[a].y * h))
                pt2 = (int(lms[b].x * w), int(lms[b].y * h))
                color = self.CONNECTION_COLORS.get((a, b), (200, 200, 200))
                cv2.line(frame, pt1, pt2, color, 3)
        
        # Draw joint points (skip face 0-10, skip hand extras 17-22)
        draw_indices = set(range(11, 17)) | set(range(23, 29))
        for i in draw_indices:
            if lms[i].visibility > 0.5:
                pt = (int(lms[i].x * w), int(lms[i].y * h))
                cv2.circle(frame, pt, 5, (255, 255, 255), -1)
                cv2.circle(frame, pt, 5, (0, 0, 0), 1)
    
    def close(self):
        """Release all resources."""
        self.camera.release()
        self.pose.close()
        cv2.destroyAllWindows()
