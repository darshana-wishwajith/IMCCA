"""
input_controller.py (MODULE 3)
Controllers via VGamepad

Purpose:
- Send Xbox 360 controller inputs directly to the system.
- Supports Buttons (taps/holds) and simulated Joysticks (left/right).
"""

import vgamepad as vg
import time

class InputController:
    def __init__(self):
        # Initialize virtual Xbox 360 gamepad
        self.gamepad = vg.VX360Gamepad()
        
        # Xbox 360 Mapping
        self.mapping = {
            # Punches -> Face Buttons
            "PUNCH_RIGHT": {"btn": vg.XUSB_BUTTON.XUSB_GAMEPAD_X, "t": "tap"},
            "KICK_RIGHT":  {"btn": vg.XUSB_BUTTON.XUSB_GAMEPAD_A, "t": "tap"},
            
            # Defense -> Triggers/Shoulders
            "BLOCK":       {"btn": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER, "t": "hold"},
            
            # Mobility -> Left Joystick (handled directly)
            # Jump/Crouch can also be mapped if needed, or handled via Joystick Y.
        }
        self.held = set()
        
        # Joystick internal state to prevent spamming USB reports if no change
        self.last_j_x = 0.0
        self.last_j_y = 0.0

    def update_joystick(self, x_val, y_val):
        """
        Receives normalized values (-1.0 to 1.0).
        Maps to both Analog Left Joystick AND the D-PAD (best for fighting games like MK11).
        """
        # Small deadzone to prevent jitter
        if abs(x_val) < 0.1: x_val = 0.0
        if abs(y_val) < 0.1: y_val = 0.0
        
        # Only update if the value changed enough to matter
        if abs(self.last_j_x - x_val) > 0.05 or abs(self.last_j_y - y_val) > 0.05:
            
            # 1. Analog Joystick Mapping
            self.gamepad.left_joystick_float(x_value_float=x_val, y_value_float=y_val)
            
            # 2. D-PAD Mapping (Crucial for Mortal Kombat 11)
            # Reset all DPAD buttons first
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
            
            # Press appropriate DPAD buttons
            if y_val > 0.5:
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            elif y_val < -0.5:
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
                
            if x_val < -0.5:
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
            elif x_val > 0.5:
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
            
            self.gamepad.update()
            
            self.last_j_x = x_val
            self.last_j_y = y_val

    def handle_actions(self, actions):
        a_set = set(actions)
        changed = False

        # Taps
        for a in a_set:
            if a in self.mapping and self.mapping[a]["t"] == "tap":
                self.gamepad.press_button(button=self.mapping[a]["btn"])
                changed = True

        # Holds
        for a in a_set:
            if a in self.mapping and self.mapping[a]["t"] == "hold" and a not in self.held:
                self.gamepad.press_button(button=self.mapping[a]["btn"])
                self.held.add(a)
                changed = True

        # Release logic for holds
        released = [a for a in self.held if a not in a_set]
        for a in released:
            self.gamepad.release_button(button=self.mapping[a]["btn"])
            self.held.remove(a)
            changed = True

        if changed:
            self.gamepad.update()
            
        # Release taps on the very next frame so they don't get stuck
        taps = [a for a in a_set if a in self.mapping and self.mapping[a]["t"] == "tap"]
        if len(taps) > 0:
            time.sleep(0.05) # Tiny sleep to ensure game registers brief tap
            for a in taps:
                self.gamepad.release_button(button=self.mapping[a]["btn"])
            self.gamepad.update()

    def release_all(self):
        for a in list(self.held):
            self.gamepad.release_button(button=self.mapping[a]["btn"])
        self.held.clear()
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()
