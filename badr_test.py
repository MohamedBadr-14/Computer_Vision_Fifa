from computer_vision import ComputerVision
import cv2



# frames = comp.read_frames("./badr/all")
# frames = [cv2.imread("off.png")]
# comp.run(frames, attempts=1)
# from computer_vision import ComputerVision
# import cv2
# from rl import DQNAgent, get_state_from_positions, calculate_reward



if __name__ == "__main__":
    # Example usage
    # video_frames = comp.read_frames("input_videos/general_small.mp4")
    # attempts = 1
    # comp.run(video_frames, attempts)

    comp = ComputerVision()
    frames = comp.read_frames("./badr/all")

    ft,hb = comp.run(frames, attempts=1)

# import keyboard

# def stop_movement_action(self, action):
#         match action:
#             case 8:
#                 keyboard.release('w')
#                 keyboard.release('d')
#             case 9:
#                 keyboard.release('w')
#                 keyboard.release('d')
#             case 10:
#                 keyboard.release('d')
#             case 11:
#                 keyboard.release('s')
#                 keyboard.release('d')
#             case 12:
#                 keyboard.release('s')

#             case 13:
#                 keyboard.release('s')
#                 keyboard.release('a')

#             case 14:
#                 keyboard.release('a')

#             case 15:
#                 keyboard.release('w')
#                 keyboard.release('a') 

# for i in range(8,16)  :
#     stop_movement_action(None, i)

