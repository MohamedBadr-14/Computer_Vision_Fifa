import time
import numpy as np
from mss import mss
import cv2
import keyboard 
from ultralytics import YOLO
import os
from RL import *

def fifa_vision():
    # Initialize model
    model = YOLO('yolov8n.pt')  

    # Setup output folder
    output_folder = "yolo_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize screen capture
    monitor = {"top": 0, "left": 0, "width": 2560, "height": 1600}  # Adjust for Legion 5 Pro
    
    frame_count = 0
    
    # Initialize mss for screen capture
    with mss() as sct:
        print("Press 's' to start capturing, 'q' to quit")
        
        # Wait for start command
        while not keyboard.is_pressed('s'):
            time.sleep(0.1)
            if keyboard.is_pressed('q'):
                print("Quitting...")
                return
        
        print("Starting capture cycle...")
        running = True
        
        while running:
            # 1. Take screenshot and pause game
            print(f"[Frame {frame_count}] Taking screenshot...")
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # keyboard.press_and_release('esc')  # Pause game
            keyboard.press('esc')
            time.sleep(0.1)
            keyboard.release('esc')
            print("Game paused, processing image...")
            # computer vision out all_tracks
            



            



            
            # 2. Process with YOLO
            # results = model.predict(frame, save=False)
            # annotated_frame = results[0].plot()
            
            # # 3. Save the output image
            # output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            # cv2.imwrite(output_path, annotated_frame)
            # print(f"Image processed and saved to {output_path}")
            
            # 4. Wait for processing to complete
            time.sleep(2)  # Simulate processing time




            
            # 5. Unpause the game
            # keyboard.press_and_release('esc')  # Unpause game
            keyboard.press('esc')  # Unpause game
            time.sleep(0.1)
            keyboard.release('esc')
            print("Game unpaused")
            
            
            # 6. Wait before next capture
            time.sleep(1)  # Wait 1 second before next capture
            
            # 7. Increment frame counter
            frame_count += 1
            
            # Check for quit command
            if keyboard.is_pressed('q'):
                print("Quit command detected, stopping...")
                running = False
    
    print("FIFA Vision has stopped.")

if __name__ == "__main__":
    fifa_vision()