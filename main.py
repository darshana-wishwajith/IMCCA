"""
main.py - IMCCA Project Entry Point
Final Clean Version

Controls:
- Perform motions (Punch, Kick, Block, etc.) to control the game.
- M to toggle SWIPE / PROPORTIONAL joystick mode.
- ESC to quit.
- Calibration happens automatically at startup.
"""

import cv2
import time
from body_skeleton import FullBodySkeleton, CameraConfig, PoseConfig, GlobalConfig
from motion_detector import MotionDetector
from swipe_detector import SwipeMotionDetector
from input_controller import InputController

def draw_ui(frame, actions, detector, fps, is_active, joy_x=0.0, joy_y=0.0, h_list=None, mode="swipe"):
    h, w = frame.shape[:2]
    
    # --- SLIM SIDEBAR (right side) ---
    sw = 160
    ov = frame.copy()
    cv2.rectangle(ov, (w-sw, 0), (w, h), (20,20,20), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    x0 = w - sw + 10
    c_t, c_c = ("LIVE", (0,255,0)) if is_active else ("IDLE", (0,0,255))
    cv2.putText(frame, c_t, (x0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_c, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (x0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    # Mode indicator
    if mode == "swipe":
        cv2.putText(frame, "SWIPE", (x0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,128), 2)
        # Left hand: fist/open state
        is_open = getattr(detector, '_hand_is_open', False)
        fcount = getattr(detector, '_finger_count', 0)
        if is_open:
            cv2.putText(frame, f"LH OPEN({fcount})", (x0, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
        else:
            cv2.putText(frame, f"LH FIST({fcount})", (x0, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
        # Left hand swipe hold state
        s_state = getattr(detector, 'swipe_state', 'IDLE')
        s_clr = (0,255,255) if 'HOLD' in s_state else (150,150,150)
        cv2.putText(frame, s_state, (x0, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.3, s_clr, 1)
        # Right hand: attack state
        rh = getattr(detector, 'rh_state', '')
        if rh == "PUNCH":
            cv2.putText(frame, "RH: PUNCH!", (x0, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 2)
        elif rh == "KICK":
            cv2.putText(frame, "RH: KICK!", (x0, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 2)
        elif rh == "BLOCK":
            cv2.putText(frame, "RH: BLOCK", (x0, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 2)
        else:
            cv2.putText(frame, f"RH: idle", (x0, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)
    else:
        cv2.putText(frame, "POSITION", (x0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 2)
        # Hand tracking status
        hand_ok = getattr(detector, '_neutral_x', None) is not None
        h_lbl, h_clr = ("TRACKING", (0,255,0)) if hand_ok else ("NO HAND", (0,0,255))
        cv2.putText(frame, h_lbl, (x0, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, h_clr, 1)
    
    cv2.putText(frame, "M=toggle", (x0, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
    cv2.putText(frame, f"X:{joy_x:+.2f}", (x0, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    cv2.putText(frame, f"Y:{joy_y:+.2f}", (x0, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    dirs = []
    if joy_y > 0.15: dirs.append("UP")
    if joy_y < -0.15: dirs.append("DN")
    if joy_x < -0.15: dirs.append("L")
    if joy_x > 0.15: dirs.append("R")
    cv2.putText(frame, "-".join(dirs) if dirs else "CENTER", (x0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    sens = getattr(detector, 'JOY_MAX_REACH', 0.06)
    cv2.putText(frame, f"Sens: {sens:.3f}", (x0, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.putText(frame, "+/- keys", (x0, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
    
    # --- DRAW HAND LANDMARKS ---
    if h_list:
        for hand in h_list:
            if hand["label"] == "Left":
                # Yellow circle on left hand palm center
                pcx, pcy = detector._get_palm_center(hand["landmarks"])
                if pcx is not None:
                    cv2.circle(frame, (int(pcx*w), int(pcy*h)), 8, (0,255,255), 2)
                    cv2.circle(frame, (int(pcx*w), int(pcy*h)), 3, (0,255,255), -1)
            elif hand["label"] == "Right" and mode == "swipe":
                # Green circle on right hand palm center
                pcx, pcy = detector._get_palm_center(hand["landmarks"])
                if pcx is not None:
                    cv2.circle(frame, (int(pcx*w), int(pcy*h)), 8, (0,255,0), 2)
                    cv2.circle(frame, (int(pcx*w), int(pcy*h)), 3, (0,255,0), -1)
    
    # --- VIRTUAL JOYSTICK (bottom-left) ---
    r_px = 50
    gap = 25
    cx, cy = r_px + gap + 20, h - r_px - gap - 40
    ov2 = frame.copy()
    cv2.circle(ov2, (cx, cy), r_px + gap, (0,0,0), -1) # Black margin
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    cv2.circle(frame, (cx, cy), r_px, (120,120,120), 2)
    cv2.line(frame, (cx-r_px,cy), (cx+r_px,cy), (60,60,60), 1)
    cv2.line(frame, (cx,cy-r_px), (cx,cy+r_px), (60,60,60), 1)
    cv2.circle(frame, (cx, cy), 3, (200,200,200), -1)
    
    # Active directions (can be multiple for diagonals)
    active_dirs = set()
    if joy_y > 0.15: active_dirs.add("UP")
    elif joy_y < -0.15: active_dirs.add("DN")
    if joy_x < -0.15: active_dirs.add("L")
    elif joy_x > 0.15: active_dirs.add("R")
        
    # 4 labels with spacing (Black inactive, Yellow active)
    loff = r_px + gap - 5
    for lbl, pos in [("UP",(cx-10,cy-loff)),("DN",(cx-10,cy+loff+10)),("L",(cx-loff-12,cy+5)),("R",(cx+loff-5,cy+5))]:
        clr = (0,255,255) if lbl in active_dirs else (0,0,0)
        thick = 2 if lbl in active_dirs else 1
        cv2.putText(frame, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, thick)
        
    # Yellow dot
    jx, jy = int(cx + joy_x*r_px), int(cy - joy_y*r_px)
    if abs(joy_x) > 0.01 or abs(joy_y) > 0.01:
        cv2.line(frame, (cx,cy), (jx,jy), (0,200,200), 2)
    cv2.circle(frame, (jx, jy), 8, (0,255,255), -1)
    
    # Sensitivity display on the left panel (below joystick)
    sens = getattr(detector, 'JOY_MAX_REACH', 0.06)
    cv2.putText(frame, f"Sensitivity: {sens:.3f} (+/- adjust)", (20, cy + r_px + gap + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
def main():
    try:
        from body_skeleton import HandsConfig
        sk = FullBodySkeleton(
            CameraConfig(mirror=True), 
            PoseConfig(model_complexity=0), 
            HandsConfig(max_num_hands=2),
            global_cfg=GlobalConfig(processor="GPU")
        )
        det_prop  = MotionDetector()       # Original proportional mode
        det_swipe = SwipeMotionDetector()   # New swipe mode
        use_swipe = True                    # <<< Start with swipe mode (new default)
        det = det_swipe
        ctrl = InputController()
    except Exception as e: print(f"Initialization Failed: {e}"); return
    
    mode_label = "SWIPE" if use_swipe else "POSITION"
    print(f"IMCCA Started. Mode: {mode_label}  |  Press M to toggle mode.")
    print("Gestures: 10-Fingers=START, Index+Middle Only=STOP. ESC to quit.")
    prev = time.time()
    is_active = False
    
    while True:
        try:
            fr = sk.read_frame()
            if fr is None: break
            p_res, h_res = sk.infer(fr); p_dict = sk.get_pose_landmarks_dict(p_res)
            h_list = sk.get_hand_landmarks_list(h_res) 
            
            # Check for Start/Stop Gestures (Finger Count)
            gesture = det.check_gestures(h_list)
            if gesture == "START":
                if not is_active:
                    print(">>> CAPTURING STARTED")
                    is_active = True
                    det.calibrate(p_dict, h_list) # Instant Auto-Calibration on startup
            elif gesture == "STOP":
                if is_active: print(">>> CAPTURING STOPPED"); is_active = False

            acts = []
            joy_x, joy_y = 0.0, 0.0
            if is_active:
                acts, joy_x, joy_y = det.detect(p_dict, h_list)
                ctrl.handle_actions(acts)
                ctrl.update_joystick(joy_x, joy_y)
                
                # Log attacks to terminal
                for a in acts:
                    if a in ("PUNCH_RIGHT", "KICK_RIGHT"):
                        print(f">>> ATTACK: {a}")
                
                # Print joystick direction to terminal (throttled)
                if abs(joy_x) > 0.01 or abs(joy_y) > 0.01:
                    _now = time.time()
                    if not hasattr(det, '_last_joy_print') or _now - det._last_joy_print > 0.2:
                        dirs = []
                        if joy_y > 0.15: dirs.append("UP")
                        if joy_y < -0.15: dirs.append("DOWN")
                        if joy_x < -0.15: dirs.append("LEFT")
                        if joy_x > 0.15: dirs.append("RIGHT")
                        if dirs:
                            print(f"JOY: {'-'.join(dirs)}  (X:{joy_x:+.2f} Y:{joy_y:+.2f})")
                        det._last_joy_print = _now
                
            sk.draw_full_skeleton(fr, p_res, h_res)
            now = time.time(); fps = 1.0 / (now - prev + 1e-6); prev = now
            cur_mode = "swipe" if use_swipe else "position"
            draw_ui(fr, acts, det, fps, is_active, joy_x, joy_y, h_list, mode=cur_mode)
            cv2.imshow("IMCCA - Immersive Motion Combat Arena", fr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break  # ESC
            
            # --- MODE TOGGLE (M key) ---
            elif key == ord('m') or key == ord('M'):
                use_swipe = not use_swipe
                det = det_swipe if use_swipe else det_prop
                mode_label = "SWIPE" if use_swipe else "POSITION"
                print(f">>> Mode switched to: {mode_label}")
                # Re-calibrate the new detector with current pose
                if is_active and p_dict:
                    det.calibrate(p_dict, h_list)
                ctrl.release_all()  # Release any held inputs during switch
            
            # --- SENSITIVITY (+/- keys, mode-aware) ---
            elif key == ord('+') or key == ord('='):
                if use_swipe:
                    det.ACTIVATION_DISPLACEMENT = max(0.01, det.ACTIVATION_DISPLACEMENT - 0.005)
                    det.NEUTRAL_ZONE_RADIUS = max(0.008, det.NEUTRAL_ZONE_RADIUS - 0.003)
                    print(f">>> [SWIPE] Sensitivity UP: Activ={det.ACTIVATION_DISPLACEMENT:.3f} Neutral={det.NEUTRAL_ZONE_RADIUS:.3f}")
                else:
                    det.JOY_MAX_REACH = max(0.02, det.JOY_MAX_REACH - 0.005)
                    print(f">>> Sensitivity UP: {det.JOY_MAX_REACH:.3f}")
            elif key == ord('-'):
                if use_swipe:
                    det.ACTIVATION_DISPLACEMENT = min(0.10, det.ACTIVATION_DISPLACEMENT + 0.005)
                    det.NEUTRAL_ZONE_RADIUS = min(0.06, det.NEUTRAL_ZONE_RADIUS + 0.003)
                    print(f">>> [SWIPE] Sensitivity DN: Activ={det.ACTIVATION_DISPLACEMENT:.3f} Neutral={det.NEUTRAL_ZONE_RADIUS:.3f}")
                else:
                    det.JOY_MAX_REACH = min(0.15, det.JOY_MAX_REACH + 0.005)
                    print(f">>> Sensitivity DOWN: {det.JOY_MAX_REACH:.3f}")
            
            # --- RESET NEUTRAL (R key, mode-aware) ---
            elif key == ord('r'):
                if use_swipe:
                    det.reset_neutral()
                else:
                    det._neutral_x = None
                    det._neutral_y = None
                    det._smooth_palm_x = None
                    det._smooth_palm_y = None
                    print(">>> Neutral center RESET")
            
            # --- ATTACK SENSITIVITY ([ ] keys, swipe mode only) ---
            elif key == ord(']') and use_swipe:
                det.ATTACK_DISP_THRESH = max(0.005, det.ATTACK_DISP_THRESH - 0.003)
                print(f">>> Attack sensitivity UP: {det.ATTACK_DISP_THRESH:.3f} (easier to trigger)")
            elif key == ord('[') and use_swipe:
                det.ATTACK_DISP_THRESH = min(0.06, det.ATTACK_DISP_THRESH + 0.003)
                print(f">>> Attack sensitivity DN: {det.ATTACK_DISP_THRESH:.3f} (harder to trigger)")
        except Exception as e:
            print(f"Main loop error: {e}")
            import traceback
            traceback.print_exc()
            # Continue gracefully to the next frame instead of crashing or freezing
            continue

    sk.close()
    if 'ctrl' in locals(): ctrl.release_all()

if __name__ == "__main__": main()
