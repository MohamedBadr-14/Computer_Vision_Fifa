import pygame
import time
import cv2
import numpy as np
import pyautogui
import threading
import json
from datetime import datetime
import os

class GamepadVideoMonitor:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Check for connected joysticks
        self.joystick = None
        self.previous_button_states = {}
        
        # Button mappings for our specific controls
        self.button_mappings = {
            11: "UP ARROW",
            12: "DOWN ARROW", 
            13: "LEFT ARROW",
            14: "RIGHT ARROW",
            0: "PASS (A/Cross)",
            1: "SHOOT (B/Circle)"
        }
        
        # Direction mappings for pass direction (compass style)
        self.arrow_directions = {
            11: "NORTH",
            12: "SOUTH", 
            13: "WEST",
            14: "EAST"
        }
        
        # Track current arrow direction for pass
        self.current_direction = "NORTH"  # Default direction
        self.pressed_arrows = set()  # Track multiple arrow presses for diagonal directions
        
        # Video recording setup
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.fps = 30  # Target FPS for recording
        self.frame_time = 1.0 / self.fps  # Time per frame in seconds
        
        # Keypress tracking
        self.keypress_log = []
        self.recording_start_time = None
        
        # Screen capture setup
        self.screen_size = pyautogui.size()
        
        self.connect_joystick()
    
    def connect_joystick(self):
        """Connect to the first available joystick"""
        joystick_count = pygame.joystick.get_count()
        
        if joystick_count == 0:
            print("No gamepad detected. Please connect a gamepad and restart.")
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Gamepad connected: {self.joystick.get_name()}")
        print("-" * 60)
        print("Video Recording Controls:")
        print("• Press Button 2 (X/Square) to START/STOP recording")
        print("• Recording will be saved as MP4 with keypress log")
        print("-" * 60)
        print("Game Controls:")
        print("• Arrow Keys: Buttons 11-14 (UP/DOWN/LEFT/RIGHT)")
        print("• Pass: Button 0 (A/Cross)")
        print("• Shoot: Button 1 (B/Circle)")
        print("• Press Ctrl+C to exit")
        print("-" * 60)
        
        # Initialize previous states for all buttons we care about (including record button)
        self.previous_button_states = {button: False for button in self.button_mappings.keys()}
        self.previous_button_states[2] = False  # Record button
        
        return True
    
    def start_recording(self):
        """Start video recording"""
        if self.recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = f"gamepad_recording_{timestamp}.mp4"
        self.log_filename = f"keypress_log_{timestamp}.json"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_filename, 
            fourcc, 
            self.fps, 
            self.screen_size
        )
        
        self.recording = True
        self.frame_count = 0
        self.keypress_log = []
        self.recording_start_time = time.time()
        
        print(f"🔴 RECORDING STARTED: {self.video_filename}")
        
        # Start video capture thread
        self.capture_thread = threading.Thread(target=self.capture_video, daemon=True)
        self.capture_thread.start()
    
    def stop_recording(self):
        """Stop video recording and save keypress log"""
        if not self.recording:
            return
        
        self.recording = False
        
        # Wait a moment for the capture thread to finish
        time.sleep(0.1)
        
        if self.video_writer:
            self.video_writer.release()
        
        # Save keypress log
        self.save_keypress_log()
        
        print(f"⏹️ RECORDING STOPPED")
        print(f"📹 Video saved: {self.video_filename}")
        print(f"📋 Keypress log saved: {self.log_filename}")
        print(f"📊 Total frames: {self.frame_count}")
    
    def capture_video(self):
        """Video capture loop running in separate thread"""
        last_frame_time = time.time()
        
        while self.recording:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time >= self.frame_time:
                try:
                    # Capture screen
                    screenshot = pyautogui.screenshot()
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    if self.video_writer:
                        self.video_writer.write(frame)
                        self.frame_count += 1
                    
                    last_frame_time = current_time
                
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    break
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)
    
    def log_keypress(self, button_name, action_type="PRESS", direction=None):
        """Log keypress with frame and timestamp information"""
        if not self.recording:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.recording_start_time
        
        log_entry = {
            "frame": self.frame_count,
            "timestamp": elapsed_time,
            "button": button_name,
            "action": action_type,
            "direction": direction,
            "datetime": datetime.now().isoformat()
        }
        
        self.keypress_log.append(log_entry)
    
    def save_keypress_log(self):
        """Save keypress log to JSON file"""
        log_data = {
            "video_file": self.video_filename,
            "total_frames": self.frame_count,
            "fps": self.fps,
            "duration_seconds": self.frame_count / self.fps,
            "keypresses": self.keypress_log
        }
        
        with open(self.log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def update_direction(self):
        """Update current direction based on currently pressed arrows (supports diagonals)"""
        if not self.pressed_arrows:
            return
        
        # Single direction presses
        if len(self.pressed_arrows) == 1:
            arrow = list(self.pressed_arrows)[0]
            self.current_direction = self.arrow_directions[arrow]
        
        # Diagonal directions (two arrows pressed)
        elif len(self.pressed_arrows) == 2:
            arrows = sorted(list(self.pressed_arrows))
            
            # Check for valid diagonal combinations
            if 11 in arrows and 14 in arrows:  # UP + RIGHT
                self.current_direction = "NORTHEAST"
            elif 11 in arrows and 13 in arrows:  # UP + LEFT  
                self.current_direction = "NORTHWEST"
            elif 12 in arrows and 14 in arrows:  # DOWN + RIGHT
                self.current_direction = "SOUTHEAST" 
            elif 12 in arrows and 13 in arrows:  # DOWN + LEFT
                self.current_direction = "SOUTHWEST"
            else:
                # Invalid combination, use the first arrow pressed
                self.current_direction = self.arrow_directions[arrows[0]]
    
    def monitor_input(self):
        """Main monitoring loop"""
        if not self.joystick:
            return
        
        try:
            while True:
                # Process all pygame events to prevent buffer overflow
                pygame.event.pump()
                
                # Check record button (button 2)
                if self.joystick.get_numbuttons() > 2:
                    current_record_state = self.joystick.get_button(2)
                    previous_record_state = self.previous_button_states[2]
                    
                    if current_record_state and not previous_record_state:
                        if not self.recording:
                            self.start_recording()
                        else:
                            self.stop_recording()
                    
                    self.previous_button_states[2] = current_record_state
                
                # Check only the buttons we care about
                for button_id in self.button_mappings.keys():
                    if button_id < self.joystick.get_numbuttons():  # Safety check
                        current_state = self.joystick.get_button(button_id)
                        previous_state = self.previous_button_states[button_id]
                        
                        # Detect button press (transition from False to True)
                        if current_state and not previous_state:
                            button_name = self.button_mappings[button_id]
                            
                            # Handle arrow buttons - add to pressed set and update direction
                            if button_id in [11, 12, 13, 14]:
                                self.pressed_arrows.add(button_id)
                                self.update_direction()
                                print(f"🎯 {button_name} - Direction set to: {self.current_direction}")
                                self.log_keypress(button_name, "PRESS", self.current_direction)
                            
                            # Handle Pass button - show direction
                            elif button_id == 0:
                                action_msg = f"PASS {self.current_direction}"
                                print(f"⚽ {action_msg}")
                                self.log_keypress("PASS", "PRESS", self.current_direction)
                            
                            # Handle Shoot button
                            elif button_id == 1:
                                print(f"🔫 SHOOT")
                                self.log_keypress("SHOOT", "PRESS")
                        
                        # Detect button release - remove from pressed set
                        elif not current_state and previous_state:
                            if button_id in [11, 12, 13, 14]:
                                button_name = self.button_mappings[button_id]
                                self.pressed_arrows.discard(button_id)
                                self.log_keypress(button_name, "RELEASE", self.current_direction)
                                
                                if self.pressed_arrows:  # If other arrows still pressed
                                    self.update_direction()
                        
                        # Update previous state
                        self.previous_button_states[button_id] = current_state
                
                # Reduced delay for better responsiveness
                time.sleep(0.005)  # 5ms
                
        except KeyboardInterrupt:
            print("\n" + "-" * 60)
            if self.recording:
                self.stop_recording()
            print("Monitoring stopped.")
        finally:
            if self.recording:
                self.stop_recording()
            pygame.quit()
            cv2.destroyAllWindows()

def main():
    print("Enhanced Gamepad Monitor with Video Recording")
    print("=" * 60)
    
    # Check if required packages are available
    try:
        import cv2
        import pyautogui
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nInstall required packages with:")
        print("pip install pygame opencv-python pyautogui")
        return
    
    monitor = GamepadVideoMonitor()
    if monitor.joystick:
        monitor.monitor_input()

if __name__ == "__main__":
    print("Required packages: pygame, opencv-python, pyautogui")
    print("Install with: pip install pygame opencv-python pyautogui")
    print("=" * 60)
    
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all packages are installed and a gamepad is connected.")