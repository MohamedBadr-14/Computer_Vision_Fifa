import os
import keyboard  # Install with: pip install keyboard
import time

action_list = {
    0: "pass_N",
    1: "pass_NE",
    2: "pass_E",
    3: "pass_SE",
    4: "pass_S",
    5: "pass_SW",
    6: "pass_W",
    7: "pass_NW",
    8: "dribble_N",
    9: "dribble_NE",
    10: "dribble_E",
    11: "dribble_SE",
    12: "dribble_S",
    13: "dribble_SW",
    14: "dribble_W",
    15: "dribble_NW",
    16: "shoot",
}


# Configuration
source_dir = r'D:\fifa-vision\offline_attempts\att3_goal'  # Replace with your source directory
rename_key = 'm'  # The key to trigger the rename
frame_count = 526  # Start from frame_1
file_extension = '.jpg'  # Adjust as needed (e.g., '.png')
new_prefix = 'renamed_'  # The new prefix to apply (you can customize this)

print(f"Waiting for key '{rename_key}' to rename files... Use right arrow to increment frame. (Press Esc to exit)")

try:
    while True:
        if keyboard.is_pressed(rename_key):
            user_input = input(f"Enter a number for frame_{frame_count}: ")
            action_num = int(user_input)
            action_name = action_list.get(action_num, "unknown_action")
            
            src_file = os.path.join(source_dir, f"frame_000{frame_count}{file_extension}")
            new_file = os.path.join(source_dir, f"frame_000{frame_count}_{action_name}{file_extension}")

            if os.path.exists(src_file):
                os.rename(src_file, new_file)
                print(f"Renamed {src_file} to {new_file}")
                # frame_count += 1  # Increment to next frame after renaming
            else:
                print(f"File not found: {src_file}")

            # Wait for key release (debounce)
            while keyboard.is_pressed(rename_key):
                pass

        if keyboard.is_pressed('right'):
            frame_count += 1
            print(f"Frame count incremented to: {frame_count}")

            # Wait for key release (debounce)
            while keyboard.is_pressed('right'):
                pass
            
        if keyboard.is_pressed('left'):
            frame_count -= 1
            print(f"Frame count incremented to: {frame_count}")

            # Wait for key release (debounce)
            while keyboard.is_pressed('left'):
                pass

        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        time.sleep(0.01)  # Add small sleep to reduce CPU usage

except KeyboardInterrupt:
    print("\nScript interrupted. Exiting...")
