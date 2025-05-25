import time
import vgamepad as vg

# Create a virtual DualShock 4 controller
gamepad = vg.VDS4Gamepad()
time.sleep(1)  # Give Windows time to register the controller

# Function to move in a direction and then return to neutral
def move_direction(direction, duration=0.5, pause=0.5):
    if direction == "up":
        gamepad.left_joystick(x_value=0, y_value=32767)  # Up (max positive y)
        print("Moving up")
    elif direction == "down":
        gamepad.left_joystick(x_value=0, y_value=-32767)  # Down (max negative y)
        print("Moving down")
    elif direction == "left":
        gamepad.left_joystick(x_value=-32767, y_value=0)  # Left (max negative x)
        print("Moving left")
    elif direction == "right":
        gamepad.left_joystick(x_value=32767, y_value=0)  # Right (max positive x)
        print("Moving right")
    
    gamepad.update()
    time.sleep(duration)  # Hold direction
    
    # Return to neutral position
    gamepad.left_joystick(x_value=0, y_value=0)
    gamepad.update()
    time.sleep(pause)  # Pause between movements
    print("Back to neutral")

print("Starting directional movements...")
# gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CROSS)   # press X
# gamepad.update()
# time.sleep(0.1)

# gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CROSS) # release X
# gamepad.update()

# Demonstrate moving in all four directions
# move_direction("up")
# move_direction("down")
# move_direction("left")
# move_direction("right")

# # You can also create diagonal movements by using both x and y values
# print("Moving diagonally (up-right)")
# gamepad.left_joystick(x_value=32767, y_value=32767)  # Up-right diagonal
# gamepad.update()
# time.sleep(0.5)
# gamepad.left_joystick(x_value=0, y_value=0)  # Return to neutral
# gamepad.update()

# print("Movement demonstration complete")
# gamepad.left_joystick_float(x_value=0, y_value=2500)  # Neutral position
# gamepad.update()

gamepad.directional_pad(vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_EAST)  # Up
gamepad.update()
time.sleep(0.2)
gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)  # Press Circle button
gamepad.update()
time.sleep(0.225)
gamepad.release_button(button=vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)
gamepad.update()
time.sleep(0.2)




# gamepad.directional_pad(vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST)  # Up-right diagonal
# gamepad.update()

# for i in range(3000):
#     print(i)
# gamepad.directional_pad(vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST)  # Up-right diagonal
# gamepad.update()
# time.sleep(0.5)  # Hold direction

# # Return to neutral position
# gamepad.directional_pad(vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST)
# gamepad.update()
# time.sleep(0.5)