import cv2
import numpy as np
from mss import mss
import time
import os
from datetime import datetime
import keyboard
class SecondDisplayRecorder:
    def __init__(self, fps=30, output_dir="recordings"):
        self.fps = fps
        self.output_dir = output_dir
        self.sct = mss()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get monitor information
        self.monitors = self.sct.monitors
        print(f"Found {len(self.monitors)} monitors:")
        for i, monitor in enumerate(self.monitors):
            print(f"Monitor {i}: {monitor}")
    
    def get_second_monitor(self):
        """Get the second monitor (index 2, since index 0 is all monitors combined)"""
        if len(self.monitors) < 3:  # Need at least 3 (all, primary, secondary)
            raise Exception("Second monitor not found. Make sure you have a second display connected.")
        return self.monitors[2]  # Index 2 is typically the second monitor
    
    def record_screen(self, duration_seconds=None, output_filename=None):
        """
        Record the second display
        
        Args:
            duration_seconds: How long to record (None for manual stop with 'q')
            output_filename: Custom filename (None for auto-generated timestamp)
        """
        try:
            monitor = self.get_second_monitor()
            print(f"Recording second monitor: {monitor}")
            
            # Generate filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"screen_recording_{timestamp}.mp4"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Get monitor dimensions
            width = monitor["width"]
            height = monitor["height"]
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            print(f"Recording started. Output: {output_path}")
            print(f"Resolution: {width}x{height} at {self.fps} FPS")
            
            if duration_seconds:
                print(f"Recording for {duration_seconds} seconds...")
            else:
                print("Press 'q' to stop recording...")
            
            start_time = time.time()
            frame_count = 0
            
            while True:
                # Capture the screen
                screenshot = self.sct.grab(monitor)
                
                # Convert to numpy array
                frame = np.array(screenshot)
                
                # Convert from BGRA to BGR (remove alpha channel)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Write the frame
                out.write(frame)
                frame_count += 1
                
                # Check for duration-based stop
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                
                # Check for manual stop (optional - requires cv2.imshow)
                # Uncomment the next 4 lines if you want to see preview and manual stop
                # cv2.imshow('Recording (Press q to stop)', cv2.resize(frame, (800, 600)))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                if keyboard.is_pressed('q'):
                    print("Manual stop requested.")
                    break
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
            
            # Release everything
            out.release()
            # cv2.destroyAllWindows()
            
            duration = time.time() - start_time
            print(f"\nRecording completed!")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Frames recorded: {frame_count}")
            print(f"File saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error during recording: {e}")
            return None

def extract_frames_from_video(video_path, output_dir=None, frame_interval=1):
    """
    Extract frames from the recorded video
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames (None for auto-generated)
        frame_interval: Extract every Nth frame (1 = all frames)
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Create output directory for frames
    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"frames_{video_name}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from: {video_path}")
    print(f"Output directory: {output_dir}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {frame_count} total frames")

# Example usage
if __name__ == "__main__":
    # Create recorder instance
    recorder = SecondDisplayRecorder(fps=30)
    
    # Record for 10 seconds (or remove duration_seconds for manual stop)
    video_path = recorder.record_screen()
    
    # Extract frames from the recorded video
    if video_path:
        extract_frames_from_video(video_path, frame_interval=1)  # Extract every 30th frameq