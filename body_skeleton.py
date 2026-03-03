"""
body_skeleton.py (MODULE 1)
Final Clean Version

Purpose:
- Capture webcam video and run MediaPipe Pose/Hands tracking.
- Optimized skeleton drawing (no face, fixed hand connectivity).
- Provides FullBodySkeleton class for use in main.py.
"""

import time
from dataclasses import dataclass
import cv2
import mediapipe as mp

@dataclass
class CameraConfig:
    camera_index: int = 0
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30
    mirror: bool = True

@dataclass
class PoseConfig:
    model_complexity: int = 0
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

@dataclass
class HandsConfig:
    max_num_hands: int = 2
    model_complexity: int = 0
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

@dataclass
class GlobalConfig:
    processor: str = "CPU" 
    hands_every_n_frames: int = 2

class FullBodySkeleton:
    def __init__(
        self,
        cam_cfg: CameraConfig = CameraConfig(),
        pose_cfg: PoseConfig = PoseConfig(),
        hands_cfg: HandsConfig = HandsConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ):
        self.cam_cfg = cam_cfg
        self.pose_cfg = pose_cfg
        self.hands_cfg = hands_cfg
        self.global_cfg = global_cfg

        if self.global_cfg.processor.upper() == "GPU":
            print("INFO: GPU selected. Note that MediaPipe Solutions (Legacy) primarily runs on CPU on Windows.")

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.pose_cfg.model_complexity,
            smooth_landmarks=self.pose_cfg.smooth_landmarks,
            min_detection_confidence=self.pose_cfg.min_detection_confidence,
            min_tracking_confidence=self.pose_cfg.min_tracking_confidence,
        )

        self.hands = None
        if self.hands_cfg.max_num_hands > 0:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.hands_cfg.max_num_hands,
                model_complexity=self.hands_cfg.model_complexity,
                min_detection_confidence=self.hands_cfg.min_detection_confidence,
                min_tracking_confidence=self.hands_cfg.min_tracking_confidence,
            )

        self.cap = cv2.VideoCapture(self.cam_cfg.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Could not open webcam.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_cfg.capture_fps)

        self.prev_time = time.time()

        # Connections & Styles
        self.FACE_LMS = set(range(0, 11))
        self.POSE_HAND_EXTRA_LMS = {17, 18, 19, 20, 21, 22}
        self.EXCLUDE_POSE_LMS = self.FACE_LMS | self.POSE_HAND_EXTRA_LMS

        self.STYLE_TORSO = {"color": (0, 255, 0), "thickness": 2}
        self.STYLE_L_ARM = {"color": (255, 0, 0), "thickness": 2}
        self.STYLE_R_ARM = {"color": (0, 0, 255), "thickness": 2}
        self.STYLE_L_LEG = {"color": (0, 255, 255), "thickness": 2}
        self.STYLE_R_LEG = {"color": (255, 0, 255), "thickness": 2}

        self.CONN_TORSO = [(11, 12), (11, 23), (12, 24), (23, 24)]
        self.CONN_LEFT_ARM = [(11, 13), (13, 15)]
        self.CONN_RIGHT_ARM = [(12, 14), (14, 16)]
        self.CONN_LEFT_LEG = [(23, 25), (25, 27), (27, 29), (29, 31)]
        self.CONN_RIGHT_LEG = [(24, 26), (26, 28), (28, 30), (30, 32)]

        self.HAND_BASE_LMS = [1, 5, 9, 13, 17]
        self.PALM_CONNECTIONS = [(5, 9), (9, 13), (13, 17)]

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok: return None
        return cv2.flip(frame, 1) if self.cam_cfg.mirror else frame

    def infer(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        p_res = self.pose.process(rgb)
        
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        if self.hands:
            if self._frame_count % self.global_cfg.hands_every_n_frames == 1 or not hasattr(self, "_last_hands_res"):
                self._last_hands_res = self.hands.process(rgb)
            h_res = getattr(self, "_last_hands_res", None)
        else:
            h_res = None
        return p_res, h_res

    def get_pose_landmarks_dict(self, pose_results):
        if not pose_results.pose_landmarks: return {}
        return {i: (lm.x, lm.y, lm.z, lm.visibility) for i, lm in enumerate(pose_results.pose_landmarks.landmark)}

    def get_hand_landmarks_list(self, hands_results):
        hands_data = []
        if not hands_results or not hands_results.multi_hand_landmarks: return hands_data
        for i, h_lms in enumerate(hands_results.multi_hand_landmarks):
            lbl = hands_results.multi_handedness[i].classification[0].label
            score = hands_results.multi_handedness[i].classification[0].score
            lms_dict = {j: (lm.x, lm.y, lm.z) for j, lm in enumerate(h_lms.landmark)}
            hands_data.append({"label": lbl, "score": score, "landmarks": lms_dict})
        return hands_data

    @staticmethod
    def _lm_to_px(lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def draw_full_skeleton(self, frame, p_res, h_res):
        if p_res.pose_landmarks:
            lms = p_res.pose_landmarks.landmark
            h_f, w_f = frame.shape[:2]
            # Draw Connections
            for grp, style in [(self.CONN_TORSO, self.STYLE_TORSO), (self.CONN_LEFT_ARM, self.STYLE_L_ARM), 
                               (self.CONN_RIGHT_ARM, self.STYLE_R_ARM), (self.CONN_LEFT_LEG, self.STYLE_L_LEG), 
                               (self.CONN_RIGHT_LEG, self.STYLE_R_LEG)]:
                for a, b in grp:
                    cv2.line(frame, self._lm_to_px(lms[a], w_f, h_f), self._lm_to_px(lms[b], w_f, h_f), 
                             style["color"], style["thickness"])
            # Draw Points
            for i in range(33):
                if i not in self.EXCLUDE_POSE_LMS:
                    cv2.circle(frame, self._lm_to_px(lms[i], w_f, h_f), 4, (200, 200, 200), -1)

        if h_res and h_res.multi_hand_landmarks:
            h_f, w_f = frame.shape[:2]
            for hand_lms in h_res.multi_hand_landmarks:
                lms = hand_lms.landmark
                # Draw standard hand connections (excluding wrist 0)
                for a, b in self.mp_hands.HAND_CONNECTIONS:
                    if a != 0 and b != 0:
                        cv2.line(frame, self._lm_to_px(lms[a], w_f, h_f), self._lm_to_px(lms[b], w_f, h_f), (0,0,0), 2)
                # Draw palm
                for a, b in self.PALM_CONNECTIONS:
                    cv2.line(frame, self._lm_to_px(lms[a], w_f, h_f), self._lm_to_px(lms[b], w_f, h_f), (0,0,0), 2)
                # Draw hand points
                for i in range(1, 21):
                    cv2.circle(frame, self._lm_to_px(lms[i], w_f, h_f), 3, (255, 70, 0), -1)
                
                # Connect to closest pose wrist
                if p_res.pose_landmarks:
                    p_lms = p_res.pose_landmarks.landmark
                    ref_px = self._lm_to_px(lms[0], w_f, h_f)
                    
                    # Available wrists
                    available_wrists = {}
                    if "left" not in getattr(self, "_used_wrists", set()):
                        available_wrists["left"] = (self._lm_to_px(p_lms[15], w_f, h_f), self.STYLE_L_ARM)
                    if "right" not in getattr(self, "_used_wrists", set()):
                        available_wrists["right"] = (self._lm_to_px(p_lms[16], w_f, h_f), self.STYLE_R_ARM)
                    
                    best_dist = float('inf')
                    best_side = None
                    best_px = None
                    best_style = None
                    
                    for side, (w_px, style) in available_wrists.items():
                        d = (ref_px[0]-w_px[0])**2 + (ref_px[1]-w_px[1])**2
                        if d < best_dist:
                            best_dist = d
                            best_side = side
                            best_px = w_px
                            best_style = style
                            
                    if best_side:
                        cv2.line(frame, best_px, ref_px, best_style["color"], best_style["thickness"])
                        if not hasattr(self, "_used_wrists"): self._used_wrists = set()
                        self._used_wrists.add(best_side)

            # Clear used wrists for the next frame
            if hasattr(self, "_used_wrists"): self._used_wrists.clear()

    def draw_fps(self, frame):
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    def close(self):
        self.cap.release()
        self.pose.close()
        self.hands.close()
        cv2.destroyAllWindows()