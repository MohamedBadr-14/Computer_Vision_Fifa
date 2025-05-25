from computer_vision import ML_ComputerVision , ComputerVision
import cv2
import time
import os

# frames = comp.read_frames("./badr/all")
# frames = [cv2.imread("off.png")]
# comp.run(frames, attempts=1)
# from computer_vision import ComputerVision
# import cv2
# from rl import DQNAgent, get_state_from_positions, calculate_reward
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
def read_frames(frames_path , max_frames=60):
        # Read the frames from the specified path
        frames = []
        first= True
        frame_first=0
        count = 0
        for filename in sorted(os.listdir(frames_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(frames_path, filename)
                frame = cv2.imread(frame_path)
                # new_frame = cv2.resize(frame, (640, 640))
                frames.append(frame)
                if count == max_frames:
                    break
                count += 1
        return frames
    

if __name__ == "__main__":
    # Example usage
    # frames = read_frames("C:/Users/DELL/Downloads/wetransfer_frames_20250524_181705_2025-05-24_1521/frames_20250524_181705")
    # attempts = 1
    # comp.run(video_frames, attempts)
    # frames = read_video("cropped_video.mp4")
    
    ml_comp = ML_ComputerVision()
    # comp = ComputerVision()
    path = r'C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/att3_goal'
    frames = read_frames(path , max_frames=150)
    print(len(frames))
    # Create benchmark tool and run

    print(frames[0].shape)
    start = time.time()
    ft,hb = ml_comp.run(frames, attempts=1)
    print("FINISHED ML" , time.time() - start)
    # start = time.time()
    # x = comp.run(frames, attempts=1)
    # print("FINISHED CV" , time.time() - start)
    

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

