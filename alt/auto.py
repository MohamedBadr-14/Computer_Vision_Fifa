import os
import keyboard  # pip install keyboard
import time

# Map action indices to names
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

# Map physical keys to action indices
# Here: '0'→0, '1'→1, … '9'→9, 'a'→10, 'b'→11, … 'f'→15, 'g'→16 (example)
action_key_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15,
    'g': 16,
}

# Configuration
source_dir = r'D:\Computer_Vision_Fifa\dataset finall\mid_change_3\aaaaa\att_0\frames'  # Replace with your source directory
rename_key = 'm'  # The key to trigger the rename
frame_count = 1256  # Start from frame_1
file_extension = '.jpg'  # Adjust as needed (e.g., '.png')

print("Controls:")
print("  ← / → : step backward/forward one frame")
print("  0-9, a-g : rename current frame to that action + advance")
print("  Esc      : exit")

try:
    while True:
        # Frame stepping
        if keyboard.is_pressed('right'):
            frame_count += 1
            print(f"Frame → {frame_count}")
            while keyboard.is_pressed('right'):
                pass

        if keyboard.is_pressed('left'):
            if frame_count > 1:
                frame_count -= 1
                print(f"Frame ← {frame_count}")
            else:
                print("Already at frame 1")
            while keyboard.is_pressed('left'):
                pass

        # Action-key detection
        for key, action_num in action_key_map.items():
            if keyboard.is_pressed(key):
                action_name = action_list[action_num]
                frame_str = f"{frame_count:06d}"
                src = os.path.join(source_dir, f"frame_{frame_str}{file_extension}")
                dst = os.path.join(source_dir, f"frame_{frame_str}_{action_name}{file_extension}")
                if os.path.exists(src):
                    os.rename(src, dst)
                    print(f"Renamed → {action_name}, next frame")
                    # frame_count += 1
                else:
                    print(f"Missing file: {src}")
                # debounce this key
                while keyboard.is_pressed(key):
                    pass
                break  # don’t check other keys until next loop

        # Exit
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        time.sleep(0.01)

except Exception as e:
    print("\nInterrupted, exiting.")
