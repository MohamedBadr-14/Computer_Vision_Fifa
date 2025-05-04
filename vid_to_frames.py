# from mp4 to frames
import cv2
import os
import numpy as np  
import sys
import glob


def extract_frames_from_video(video_path, output_dir, start_frame=0, end_frame=None):
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
        start_frame (int): Frame number to start extraction from.
        end_frame (int): Frame number to stop extraction at. If None, extract until the end of the video.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    print(f"Total frames in video: {total_frames}")

    # Loop through the video frames
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break  # Break if no more frames are available

        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f'frame_{frame_num:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved {frame_filename}")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Example usage
    video_path = 'to_f.mp4'  # Path to your video file
    output_dir = 'frames_alllllll'  # Directory to save extracted frames
    start_frame = 0  # Start frame number
    end_frame = None  # End frame number (None means extract until the end)

    extract_frames_from_video(video_path, output_dir, start_frame, end_frame)