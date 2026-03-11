"""
input_sender.py — Keyboard Input Sender
IMCCA v2

Sends keyboard inputs to MK11 using direct key simulation.
Threaded input queue for zero-blocking key sends.
Supports:
- Tap (press + release with configurable duration)
- Hold (press and hold until released)
- Combo macros (timed key sequences)
"""

import time
import threading
from collections import deque
from typing import Set, Optional, List, Tuple
from pynput.keyboard import Controller, Key

from config import MK11_KEYS


# Map special key names to pynput Key objects
SPECIAL_KEYS = {
    "space": Key.space,
    "enter": Key.enter,
    "esc": Key.esc,
    "tab": Key.tab,
    "shift": Key.shift,
    "ctrl": Key.ctrl,
    "alt": Key.alt,
}


class InputSender:
    """
    Sends keyboard inputs for MK11 actions.
    All key presses happen in a background thread to prevent blocking.
    """
    
    def __init__(self, key_map=None):
        self.key_map = key_map or MK11_KEYS
        self.keyboard = Controller()
        
        # Currently held keys (for hold actions like BLOCK, DUCK, MOVE)
        self._held_actions: Set[str] = set()
        self._held_keys: Set = set()
        
        # Tap queue (processed by background thread)
        self._tap_queue: deque = deque()
        self._lock = threading.Lock()
        
        # Background thread for taps
        self._running = True
        self._thread = threading.Thread(target=self._tap_worker, daemon=True)
        self._thread.start()
        
        # Combo macro thread
        self._combo_thread: Optional[threading.Thread] = None
        
        # Hold actions — these keys stay pressed while the action is active
        self.HOLD_ACTIONS = {"BLOCK", "DUCK", "MOVE_LEFT", "MOVE_RIGHT", "JUMP"}
        
        # Tap actions — these keys get press+release
        self.TAP_ACTIONS = {"FRONT_PUNCH", "BACK_PUNCH", "FRONT_KICK", "BACK_KICK",
                           "THROW", "FLIP_STANCE", "CHAR_ASSIST", "FATAL_BLOW"}
        
        # Tap duration
        self.TAP_DURATION = 0.04  # 40ms — enough for game to register
    
    def _get_keys(self, action: str):
        """Convert action name to list of pynput keys (supports 'u+o' multi-key)."""
        key_name = self.key_map.get(action)
        if key_name is None:
            return []
        
        # Handle simultaneous keys like 'u+o'
        parts = key_name.split('+')
        keys = []
        for part in parts:
            part = part.strip()
            if part in SPECIAL_KEYS:
                keys.append(SPECIAL_KEYS[part])
            else:
                keys.append(part)
        return keys
    
    def handle_actions(self, actions: List[str]):
        """
        Process a list of active actions for this frame.
        - Hold actions: pressed while in the list, released when removed
        - Tap actions: single press+release
        """
        action_set = set(actions)
        
        # --- HANDLE HOLDS ---
        # Press new holds
        for action in action_set:
            if action in self.HOLD_ACTIONS and action not in self._held_actions:
                keys = self._get_keys(action)
                for key in keys:
                    try:
                        self.keyboard.press(key)
                        self._held_keys.add(key)
                    except Exception:
                        pass
                if keys:
                    self._held_actions.add(action)
        
        # Release old holds
        to_release = [a for a in self._held_actions if a not in action_set]
        for action in to_release:
            keys = self._get_keys(action)
            for key in keys:
                try:
                    self.keyboard.release(key)
                    self._held_keys.discard(key)
                except Exception:
                    pass
            self._held_actions.discard(action)
        
        # --- HANDLE TAPS ---
        for action in action_set:
            if action in self.TAP_ACTIONS:
                keys = self._get_keys(action)
                if keys:
                    with self._lock:
                        self._tap_queue.append(keys)  # Append list of keys
    
    def _tap_worker(self):
        """Background thread that processes tap key presses."""
        while self._running:
            keys = None
            with self._lock:
                if self._tap_queue:
                    keys = self._tap_queue.popleft()  # List of keys
            
            if keys:
                try:
                    # Press all keys simultaneously
                    for key in keys:
                        self.keyboard.press(key)
                    time.sleep(self.TAP_DURATION)
                    # Release all
                    for key in keys:
                        self.keyboard.release(key)
                except Exception:
                    pass
            else:
                time.sleep(0.005)  # 5ms idle polling
    
    def execute_combo(self, keys_with_delays: List[Tuple[str, float]]):
        """
        Execute a combo macro: press a sequence of keys with timing.
        Runs in a separate thread to not block detection.
        """
        def _run_combo():
            for key_name, hold_time in keys_with_delays:
                key = SPECIAL_KEYS.get(key_name, key_name)
                try:
                    self.keyboard.press(key)
                    time.sleep(hold_time)
                    self.keyboard.release(key)
                    time.sleep(0.03)  # 30ms gap between combo inputs
                except Exception:
                    pass
        
        # Only run one combo at a time
        if self._combo_thread and self._combo_thread.is_alive():
            return
        
        self._combo_thread = threading.Thread(target=_run_combo, daemon=True)
        self._combo_thread.start()
    
    def release_all(self):
        """Release all held keys."""
        for key in list(self._held_keys):
            try:
                self.keyboard.release(key)
            except Exception:
                pass
        self._held_keys.clear()
        self._held_actions.clear()
    
    def shutdown(self):
        """Clean shutdown."""
        self.release_all()
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
