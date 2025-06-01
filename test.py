# 
import keyboard as kb
import time
from controller import *
if __name__ == "__main__":
    print("Press 's' to trigger the 'p' key press and release. Press 'esc' to exit.")
    cont = Controller()
    while True:
        if kb.is_pressed('s'):
            print("S key pressed")
            kb.press_and_release('p')
        time.sleep(0.1)  # Sleep to prevent high CPU usage
                
    