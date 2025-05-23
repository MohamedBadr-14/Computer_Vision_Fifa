import pyautogui
import time
import os
from datetime import datetime
import keyboard
# Configuration
NUM_SCREENSHOTS = 200  # Number of screenshots to take before pausing
INTERVAL = 0.1  # Time interval between screenshots in seconds
SAVE_DIR = "screenshots"  # Directory to save screenshots

# Create screenshots directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def take_screenshots():
    print(f"Starting to take {NUM_SCREENSHOTS} screenshots...")
    for i in range(NUM_SCREENSHOTS):
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/screenshot_{timestamp}_{i+1}.png"
        
        # Take screenshot and save
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Saved screenshot {i+1}/{NUM_SCREENSHOTS}: {filename}")
        
        # Wait for the specified interval
        time.sleep(INTERVAL)
    
    print(f"Finished taking {NUM_SCREENSHOTS} screenshots. Pausing...")
    input("Press Enter to continue or Ctrl+C to exit...")

if __name__ == "__main__":
    #wait for key press to start
    keyboard.wait('l')
    try:
        while True:
            take_screenshots()
    except KeyboardInterrupt:
        print("\nScript terminated by user.")