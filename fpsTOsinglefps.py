import os
import cv2
import shutil
from tqdm import tqdm
from datetime import datetime

def extract_all_frames(input_video_path, output_folder):
    """
    Extract all frames from a video
    
    Args:
        input_video_path (str): Path to the input video file
        output_folder (str): Folder to save extracted frames
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    print(f"Video info:")
    print(f"  - Original FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration_seconds:.2f} seconds")
    print(f"  - Extracting ALL frames")
    
    # Frame extraction
    frame_count = 0
    
    # Use tqdm for progress bar
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save every frame
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

def main():
    # Get current timestamp for folder names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths - hardcoded input video path
    input_video = "./all.mp4"  # Fixed path, no user input
    output_folder = f"all_frames_{timestamp}"
    
    try:
        # Extract all frames
        extract_all_frames(input_video, output_folder)
        print(f"Process completed. All frames saved to: {output_folder}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()