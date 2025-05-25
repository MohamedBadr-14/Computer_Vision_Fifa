import time
import keyboard
from mss import mss
import numpy as np
import cv2
import os
import mss.tools
import time
import vgamepad as vg


class GAME:
    def __init__(self):
        """
        Initialize the GAME class with a pause frame skip value.
        
        :param pause_frames_skip: Number of frames to skip during pause.
        """
        self.skip_counter = 200
        self.pause_frames_skip = 50
        self.temp_flag = False
        self.gamepad = vg.VDS4Gamepad()
        time.sleep(1)  # Give Windows time to register the controller
        
    def pause_game(self):
        """
        Pause the game by simulating ESC key press.
        """
        self.gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_OPTIONS)  # Press Options button
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_OPTIONS)  # Release Options button
        self.gamepad.update()

    def resume_game(self,is_action=False):
        """
        Resume the game by simulating ENTER key press and reset relevant counters.
        """
        self.gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CROSS)  # Press X button
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CROSS)
        self.gamepad.update()
        self.skip_counter = self.pause_frames_skip
        self.temp_flag = False
        # if is_action:
        #     self.skip_counter = 0
        #     self.temp_flag = True
        # else:
    
    def run(self):
        """
        Main loop to handle game pause and resume.
        """
        # Fixed the mss instantiation
        with mss.mss() as sct:
            monitor = sct.monitors[2]
            print("Press 'x' to start capturing, 'q' to quit")
            
            # Wait for start command
            while not keyboard.is_pressed('x'):
                time.sleep(0.1)
            print("Game started, press 'q' to quit")
            self.gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CROSS)  # Press X button
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CROSS)  # Release X button
            self.gamepad.update()
            print("PRESS ENTER")
                
            running = True
            frame_list = []
            while running:
                # Capture the screen
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    # Process the screenshot if neededdd
                    # For example, save it or analyze it
                    
                if self.skip_counter > 0:
                    self.skip_counter -= 1
                    frame_list.append(frame)
                    continue
        
                if keyboard.is_pressed('t'):
                    self.pause_game()
                    self.temp_flag = True
                    print("Game paused")
                if keyboard.is_pressed('r'):
                    self.resume_game()
                    self.temp_flag = False
                    print("Game resumed")
                    
                # Check for quit command
                if keyboard.is_pressed('q'):
                    running = False
                    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
                    for i, frame in enumerate(frame_list):
                        # Use an absolute path to save the frames
                        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
                        print(f"Saving frame {i} to {output_dir}...")
                        cv2.imwrite(os.path.join(output_dir, f"frame_{i}.png"), frame)
                        print("Quitting the game.")
                
if __name__ == "__main__":
    game = GAME()
    game.run()