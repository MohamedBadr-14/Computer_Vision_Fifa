import numpy as np
import cv2

# Load the frames (shape: 14 batches × 5 frames × H × W × 3)
batched_frames = np.load('D:/fifa-vision/goal1/to_be_sent.npy')  # Shape: (14, 5, 1600, 2560, 3)

# Combine all batches into a single array of frames (shape: 70 × H × W × 3)
frames = batched_frames.reshape(-1, 1600, 2560, 3)  # Now shape: (70, 1600, 2560, 3)
print(f"Frames shape after reshaping: {frames.shape}")

# Video settings
output_path = 'output.mp4'
fps = 30  # Frames per second
height, width = 1600, 2560

# Initialize VideoWriter (H.264 codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write each frame
for frame in frames:
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)  # Convert [0,1] to [0,255]
    video_writer.write(frame)

video_writer.release()
print(f"Saved video to {output_path} (70 frames total)")
import numpy as np
import cv2
import os

# Load the frames (shape: 14 batches × 5 frames × H × W × 3)
batched_frames = np.load('./to_be_sent.npy')  # Shape: (14, 5, 1600, 2560, 3)

# Combine all batches into a single array of frames (shape: 70 × H × W × 3)
frames = batched_frames.reshape(-1, 1600, 2560, 3)  # Now shape: (70, 1600, 2560, 3)
print(f"Frames shape after reshaping: {frames.shape}")

# Create a directory to save the frames
output_dir = 'D:/fifa-vision/frames'  # Change this to your desired path
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

# Save each frame as an image
for i, frame in enumerate(frames):
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)  # Convert [0,1] to [0,255]
    
    # Save frame as PNG (or JPG)
    frame_filename = os.path.join(output_dir, f'frame_{i:04d}.png')  # e.g., frame_0000.png
    cv2.imwrite(frame_filename, frame)

print(f"Saved {len(frames)} frames to {output_dir}")