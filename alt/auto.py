import os
import keyboard  # Install with: pip install keyboard
import time

action_list = {
    0: "pass_N", # Basa Foo2
    1: "pass_NE", # Basa Foo2 ymeen
    2: "pass_E",  # Basa Ymeen
    3: "pass_SE", # Basa Taht ymeen
    4: "pass_S", # basa taht
    5: "pass_SW", # basa taht shmal
    6: "pass_W", # Basa Shmal
    7: "pass_NW", # Basa Foo2 shmal
    8: "dribble_N", # dribble Foo2
    9: "dribble_NE", # dribble Foo2 ymeen
    10: "dribble_E", # dribble Ymeen
    11: "dribble_SE", # dribble Taht ymeen
    12: "dribble_S", # dribble Taht
    13: "dribble_SW", # dribble Taht shmal
    14: "dribble_W", # dribble Shmal
    15: "dribble_NW", # dribble Foo2 shmal
    16: "shoot", # shoot
}


# Configuration
source_dir = r'D:\fifa-vision\offline_attempts\Att_6_goal_1\frames'  # Replace with your source directory
rename_key = 'm'  # The key to trigger the rename
frame_count = 266  # Start from frame_1
file_extension = '.jpg'  # Adjust as needed (e.g., '.png')

print(f"Waiting for key '{rename_key}' to rename files... Use right arrow to increment frame. (Press Esc to exit)")

try:
    while True:
        if keyboard.is_pressed(rename_key):
            user_input = input(f"Enter a number for frame_{frame_count}: ")
            action_num = int(user_input)
            action_name = action_list.get(action_num, "unknown_action")
            
            # Format with 6 digits (e.g., 000001, 000592)
            frame_str = f"{frame_count:06d}"
            src_file = os.path.join(source_dir, f"frame_{frame_str}{file_extension}")
            new_file = os.path.join(source_dir, f"frame_{frame_str}_{action_name}{file_extension}")

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
            if frame_count > 1:  # Prevent going below frame 1
                frame_count -= 1
                print(f"Frame count decremented to: {frame_count}")
            else:
                print("Already at first frame")

            # Wait for key release (debounce)
            while keyboard.is_pressed('left'):
                pass

        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        time.sleep(0.01)  # Add small sleep to reduce CPU usage

except KeyboardInterrupt:
    print("\nScript interrupted. Exiting...")