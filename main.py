"""
main.py — IMCCA v2 Entry Point
Immersive Motion Combat Control Arena

Exhibition-friendly Mortal Kombat 11 controller using full-body motion capture.
Uses MediaPipe Pose ONLY (no Hands model) for maximum FPS and minimum latency.

Controls:
- Raise both hands above head → START capturing
- Hands on hips → STOP capturing
- Punch with right hand → Front Punch (J)
- Punch with left hand → Back Punch (I)
- Right leg kick → Front Kick (K)
- Left leg kick → Back Kick (L)
- Cross arms on chest → Block (O)
- Both hands thrust forward → Throw (Space)
- Lean left/right → Move Left/Right (A/D)
- Jump → Jump (W)
- Crouch/squat → Duck (S)
- Quick body twist → Flip Stance (U)
- Both hands high above head (hold) → Character Assist (P)

Keyboard shortcuts:
- ESC → Quit
- R → Recalibrate
- +/- → Adjust sensitivity
"""

import cv2
import time
import numpy as np

from config import (
    CameraConfig, PoseConfig, SmoothingConfig,
    ActionThresholds, UIConfig, MK11_KEYS, DEFAULT_COMBOS
)
from pose_tracker import PoseTracker
from gesture_engine import GestureEngine
from input_sender import InputSender


# ─── ACTION DISPLAY CONFIG ────────────────────────────────────────

ACTION_COLORS = {
    "FRONT_PUNCH":  (0, 165, 255),    # Orange
    "BACK_PUNCH":   (0, 100, 255),    # Red-orange
    "FRONT_KICK":   (0, 255, 255),    # Yellow
    "BACK_KICK":    (0, 255, 180),    # Yellow-green
    "THROW":        (255, 0, 255),    # Magenta
    "BLOCK":        (255, 50, 50),    # Blue
    "MOVE_LEFT":    (255, 255, 0),    # Cyan
    "MOVE_RIGHT":   (255, 255, 0),    # Cyan
    "JUMP":         (0, 255, 0),      # Green
    "DUCK":         (0, 200, 200),    # Dark yellow
    "FLIP_STANCE":  (200, 100, 255),  # Pink
    "CHAR_ASSIST":  (255, 200, 0),    # Light blue
    "FATAL_BLOW":   (0, 0, 255),      # Bright red
}

ACTION_LABELS = {
    "FRONT_PUNCH":  "FRONT PUNCH!",
    "BACK_PUNCH":   "BACK PUNCH!",
    "FRONT_KICK":   "FRONT KICK!",
    "BACK_KICK":    "BACK KICK!",
    "THROW":        "THROW!",
    "BLOCK":        "BLOCKING",
    "MOVE_LEFT":    "< MOVE LEFT",
    "MOVE_RIGHT":   "MOVE RIGHT >",
    "JUMP":         "JUMP!",
    "DUCK":         "DUCK",
    "FLIP_STANCE":  "FLIP STANCE",
    "CHAR_ASSIST":  "CHAR ASSIST",
    "FATAL_BLOW":   ">>> FATAL BLOW <<<",
}


def draw_ui(frame, engine, fps, is_active, ui_cfg):
    """Draw the exhibition-friendly UI overlay."""
    h, w = frame.shape[:2]
    
    # ─── TOP STATUS BAR ──────────────────────────────────
    bar_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status indicator
    if is_active:
        cv2.circle(frame, (20, 20), 8, (0, 255, 0), -1)
        cv2.putText(frame, "LIVE", (35, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "IDLE — Step fully into frame to START", (35, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    
    # FPS
    fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)
    
    if not is_active:
        # Draw large instruction text when idle
        msg = "STEP FULLY INTO CAMERA FRAME"
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2
        # Background
        cv2.rectangle(frame, (tx - 20, ty - 35), (tx + text_size[0] + 20, ty + 15), (0, 0, 0), -1)
        cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "to auto-start LIVE mode!", (tx + 60, ty + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return
    
    # ─── ACTION DISPLAY (centered, large) ────────────────
    actions = engine.active_actions
    
    # Flash border on attack actions
    attack_actions = {"FRONT_PUNCH", "BACK_PUNCH", "FRONT_KICK", "BACK_KICK", "THROW", "FATAL_BLOW"}
    for action in actions:
        if action in attack_actions:
            color = ACTION_COLORS.get(action, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
            cv2.rectangle(frame, (2, 2), (w - 3, h - 3), color, 2)
    
    # Show recent attack name (large, centered, fades out)
    if engine.last_attack and (time.time() - engine.last_attack_time < 0.4):
        label = ACTION_LABELS.get(engine.last_attack, engine.last_attack)
        color = ACTION_COLORS.get(engine.last_attack, (255, 255, 255))
        
        # Large central text with shadow
        font_scale = 1.5
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2 - 60
        
        # Shadow
        cv2.putText(frame, label, (tx + 2, ty + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 4)
        # Text
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 3)
    
    # ─── ACTIVE ACTIONS LIST (right sidebar) ─────────────
    sw = ui_cfg.sidebar_width
    ov = frame.copy()
    cv2.rectangle(ov, (w - sw, bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    
    x0 = w - sw + 10
    y0 = bar_h + 25
    
    cv2.putText(frame, "ACTIVE:", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    y = y0 + 25
    for action in actions:
        if action.startswith("COMBO:"):
            label = action.replace("COMBO:", "COMBO: ")
            color = (0, 255, 255)
        else:
            label = ACTION_LABELS.get(action, action)
            color = ACTION_COLORS.get(action, (200, 200, 200))
        
        cv2.putText(frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 20
    
    if not actions:
        cv2.putText(frame, "-- none --", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    
    # Spine angle indicator
    y += 15
    angle = engine.spine_angle
    angle_color = (0, 255, 255) if angle < engine.thresh.move_angle_threshold else (150, 150, 150)
    cv2.putText(frame, f"Spine: {angle:.0f}° (thr:{engine.thresh.move_angle_threshold:.0f}°)",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, angle_color, 1)
    
    # ─── VIRTUAL JOYSTICK (bottom-left) ──────────────────
    r = 45
    gap = 20
    cx, cy = r + gap + 15, h - r - gap - 30
    
    ov2 = frame.copy()
    cv2.circle(ov2, (cx, cy), r + gap, (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.4, frame, 0.6, 0, frame)
    
    # Joystick circle
    cv2.circle(frame, (cx, cy), r, (80, 80, 80), 2)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), (40, 40, 40), 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), (40, 40, 40), 1)
    cv2.circle(frame, (cx, cy), 3, (150, 150, 150), -1)
    
    # Direction labels
    active_dirs = set()
    mx, my = engine.move_x, engine.move_y
    if mx < -0.3: active_dirs.add("L")
    if mx > 0.3: active_dirs.add("R")
    if my > 0.3: active_dirs.add("UP")
    if my < -0.3: active_dirs.add("DN")
    
    loff = r + gap - 5
    for lbl, pos in [("W", (cx - 5, cy - loff)),
                     ("S", (cx - 4, cy + loff + 10)),
                     ("A", (cx - loff - 8, cy + 5)),
                     ("D", (cx + loff - 3, cy + 5))]:
        dir_map = {"W": "UP", "S": "DN", "A": "L", "D": "R"}
        is_active_dir = dir_map[lbl] in active_dirs
        clr = (0, 255, 255) if is_active_dir else (60, 60, 60)
        thick = 2 if is_active_dir else 1
        cv2.putText(frame, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, thick)
    
    # Dot position
    jx = int(cx + mx * r)
    jy = int(cy - my * r)
    jx = max(cx - r, min(cx + r, jx))
    jy = max(cy - r, min(cy + r, jy))
    
    if abs(mx) > 0.05 or abs(my) > 0.05:
        cv2.line(frame, (cx, cy), (jx, jy), (0, 200, 200), 2)
    cv2.circle(frame, (jx, jy), 7, (0, 255, 255), -1)
    
    # ─── CONTROLS HELP (bottom-right) ────────────────────
    help_y = h - 15
    help_x = w - sw + 10
    cv2.putText(frame, "R=Recalibrate  ESC=Quit", (help_x, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    cv2.putText(frame, "+/-=Sensitivity", (help_x, help_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    
    # ─── MOTION GUIDE (bottom center) ────────────────────
    guide_y = h - 10
    guide_items = [
        "Punch R/L = Front/Back Punch",
        "Kick R/L = Front/Back Kick",
        "Arms Crossed = Block",
        "Lean = Move | Squat = Duck | Jump = Jump",
    ]
    for i, text in enumerate(guide_items):
        ty = guide_y - (len(guide_items) - 1 - i) * 16
        cv2.putText(frame, text, (140, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)


def main():
    print("=" * 60)
    print("  IMCCA v2 — Immersive Motion Combat Control Arena")
    print("  Full-Body Motion Controller for Mortal Kombat 11")
    print("=" * 60)
    print()
    
    try:
        # Initialize modules
        tracker = PoseTracker(
            cam_cfg=CameraConfig(mirror=True),
            pose_cfg=PoseConfig(model_complexity=0),
            smooth_cfg=SmoothingConfig(ema_alpha=0.6),
        )
        engine = GestureEngine(
            thresholds=ActionThresholds(),
            combos=DEFAULT_COMBOS,
        )
        sender = InputSender(key_map=MK11_KEYS)
        ui_cfg = UIConfig()
        
        print("[OK] All modules initialized.")
        print()
        print("INSTRUCTIONS:")
        print("  1. Stand facing the camera in a neutral pose")
        print("  2. Step FULLY into the frame to auto-start LIVE mode")
        print("  3. Step out of the frame or obscure body to STOP")
        print("  4. Press ESC to quit")
        print()
        
    except Exception as e:
        print(f"[FATAL] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # State
    is_active = False      # Starts idle until full body is detected
    prev_time = time.time()
    sensitivity_mult = 1.0
    calibration_done = False
    
    # Start/stop gesture debounce
    last_gesture_time = 0
    
    while True:
        try:
            frame = tracker.read_frame()
            if frame is None:
                continue
            
            # Process pose
            landmarks, raw_results = tracker.process(frame)
            
            # Check full body visibility to toggle active mode automatically
            if landmarks:
                # 11,12: Shoulders | 15,16: Wrists | 23,24: Hips | 27,28: Ankles
                key_points = [11, 12, 15, 16, 23, 24, 27, 28]
                visible_points = sum(1 for idx in key_points if idx in landmarks and landmarks[idx][3] > 0.5)
                
                # Require at least 6 out of 8 key points to be visible
                if visible_points >= 6:
                    if not is_active:
                        print(">>> Full body detected! Auto-resuming LIVE mode.")
                        is_active = True
                else:
                    if is_active:
                        print(f">>> Full body lost (vis: {visible_points}/8)! Auto-pausing to IDLE mode.")
                        is_active = False
            else:
                if is_active:
                    print(">>> No body detected! Auto-pausing to IDLE mode.")
                    is_active = False
            
            # Auto-calibrate on first valid frame (idle mode disabled)
            now = time.time()
            if not calibration_done and landmarks:
                if engine.calibrate(landmarks, tracker):
                    calibration_done = True
                    print(">>> AUTO-CALIBRATED on first frame. Fight!")
            
            # Detect and send actions
            if is_active and calibration_done:
                actions = engine.detect(landmarks, tracker)
                
                # Handle combos specially
                combo_actions = [a for a in actions if a.startswith("COMBO:")]
                regular_actions = [a for a in actions if not a.startswith("COMBO:")]
                
                # Send regular actions
                sender.handle_actions(regular_actions)
                
                # Execute combos
                for combo_action in combo_actions:
                    combo_name = combo_action.replace("COMBO:", "")
                    for combo_def in DEFAULT_COMBOS:
                        if combo_def.name == combo_name:
                            sender.execute_combo(combo_def.output_keys)
                            print(f">>> COMBO: {combo_name}")
                            break
                
                # Log attacks to console (throttled)
                for a in regular_actions:
                    if a in ("FRONT_PUNCH", "BACK_PUNCH", "FRONT_KICK", "BACK_KICK", "THROW"):
                        key = MK11_KEYS.get(a, "?")
                        print(f">>> {ACTION_LABELS.get(a, a)} → [{key.upper()}]")
            else:
                # Release everything when not active
                sender.release_all()
            
            # Draw skeleton
            tracker.draw_skeleton(frame, raw_results)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            
            # Draw UI
            draw_ui(frame, engine, fps, is_active, ui_cfg)
            
            # Show window
            cv2.imshow(ui_cfg.window_name, frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            elif key == ord('r') or key == ord('R'):
                # Recalibrate
                if landmarks:
                    engine.calibrate(landmarks, tracker)
                    print(">>> Recalibrated!")
            
            elif key == ord('+') or key == ord('='):
                # Increase sensitivity (lower attack thresholds, raise angle threshold)
                engine.thresh.punch_velocity = max(0.3, engine.thresh.punch_velocity * 0.85)
                engine.thresh.kick_velocity = max(0.3, engine.thresh.kick_velocity * 0.85)
                engine.thresh.move_angle_threshold = min(85.0, engine.thresh.move_angle_threshold + 2.0)
                print(f">>> Sensitivity UP — Punch: {engine.thresh.punch_velocity:.2f} "
                      f"Kick: {engine.thresh.kick_velocity:.2f} "
                      f"MoveAngle: {engine.thresh.move_angle_threshold:.0f}°")
            
            elif key == ord('-'):
                # Decrease sensitivity (raise attack thresholds, lower angle threshold)
                engine.thresh.punch_velocity *= 1.15
                engine.thresh.kick_velocity *= 1.15
                engine.thresh.move_angle_threshold = max(60.0, engine.thresh.move_angle_threshold - 2.0)
                print(f">>> Sensitivity DOWN — Punch: {engine.thresh.punch_velocity:.2f} "
                      f"Kick: {engine.thresh.kick_velocity:.2f} "
                      f"MoveAngle: {engine.thresh.move_angle_threshold:.0f}°")
            
        except Exception as e:
            print(f"[ERROR] Main loop: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cleanup
    print("\nShutting down...")
    sender.shutdown()
    tracker.close()
    print("IMCCA v2 terminated.")


if __name__ == "__main__":
    main()
