#read video all_frames.mp4 and crop the time from 2:35 to 2:40
import cv2
import numpy as np
import os


def crop_video(input_path, output_path, start_time, end_time):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write frames from start to end frame
    for i in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame:
            out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = 'all_frames.mp4'  # Path to your input video
    output_video_path = 'cropped_video.mp4'  # Path to save the cropped video
    start_time = 155  # Start time in seconds (2:35)
    end_time = 160  # End time in seconds (2:40)

    crop_video(input_video_path, output_video_path, start_time, end_time)