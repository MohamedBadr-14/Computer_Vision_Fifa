import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
import sys
import glob

# Import both implementations
# Ensure both implementations are in the same directory
# sys.path.append('.')
#  no module named 'ball_detect_cpu' or 'ball_detect_gpu'

from .ball_detect import BallDetectGPU
from .ball_detect_cpu import BallDetectCPU

class BenchmarkTool:
    def __init__(self):
        """Initialize the benchmark tool with both CPU and GPU implementations"""
        # Initialize both implementations
        print("Initializing CPU implementation...")
        start_time = time.time()
        self.detector_cpu = BallDetectCPU()
        self.cpu_init_time = time.time() - start_time
        
        print("Initializing GPU implementation...")
        start_time = time.time()
        self.detector_gpu = BallDetectGPU()
        self.gpu_init_time = time.time() - start_time
        
        # Stats dictionaries
        self.cpu_times = {
            'total': [],
            'mask_on_white': [],
            'find_ball': [],
            'svm_predict': []
        }
        
        self.gpu_times = {
            'total': [],
            'mask_on_white': [],
            'find_ball': [],
            'svm_predict': []
        }
        
        print(f"Initialization times - CPU: {self.cpu_init_time:.4f}s, GPU: {self.gpu_init_time:.4f}s")
        
    def run_benchmark_on_frames(self, frames):
        """Run the benchmark on pre-loaded frames
        
        Args:
            frames: List of frames to process
        """
        if not frames:
            print("Error: No frames provided")
            return
        
        num_frames = len(frames)
        print(f"Running benchmark on {num_frames} pre-loaded frames")
        
        # Benchmark CPU implementation
        print("\nRunning CPU benchmark...")
        self._benchmark_implementation(frames, self.detector_cpu, self.cpu_times)
        
        # Benchmark GPU implementation
        print("\nRunning GPU benchmark...")
        self._benchmark_implementation(frames, self.detector_gpu, self.gpu_times)
        
        # Show results
        self._show_results()
        
    def _benchmark_implementation(self, frames, detector, time_dict):
        """Run benchmark on specific implementation"""
        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frames)}")
            
            # Time the entire process
            start_total = time.time()
            
            # Time specific functions
            start_mask = time.time()
            mask = detector.mask_on_white(frame.copy())
            time_dict['mask_on_white'].append(time.time() - start_mask)
            
            start_find = time.time()
            bboxes = detector.find_ball(mask, frame.copy())
            time_dict['find_ball'].append(time.time() - start_find)
            
            # Reset detector state for fair comparison
            detector.last_ball_position = None
            
            start_predict = time.time()
            detector.svm_predict(frame.copy())
            time_dict['svm_predict'].append(time.time() - start_predict)
            
            # Record total time
            time_dict['total'].append(time.time() - start_total)
            
    def _show_results(self):
        """Display benchmark results"""
        # Calculate statistics
        cpu_avg = {k: np.mean(v) * 1000 for k, v in self.cpu_times.items()}  # ms
        gpu_avg = {k: np.mean(v) * 1000 for k, v in self.gpu_times.items()}  # ms
        
        speedup = {k: cpu_avg[k] / gpu_avg[k] if gpu_avg[k] > 0 else 0 for k in cpu_avg}
        
        # Print results in a simple table format
        headers = ["Function", "CPU Time (ms)", "GPU Time (ms)", "Speedup (x)"]
        print("\n=== Benchmark Results ===")
        
        # Try to use tabulate if available
        try:
            from tabulate import tabulate
            table_data = []
            for func in self.cpu_times.keys():
                table_data.append([
                    func, 
                    f"{cpu_avg[func]:.2f}", 
                    f"{gpu_avg[func]:.2f}", 
                    f"{speedup[func]:.2f}"
                ])
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        except ImportError:
            # Simple table format as fallback
            print(" | ".join(headers))
            print("-" * 60)
            for func in self.cpu_times.keys():
                print(f"{func:15} | {cpu_avg[func]:14.2f} | {gpu_avg[func]:14.2f} | {speedup[func]:12.2f}")
        
        # Create bar chart
        self._plot_comparison(cpu_avg, gpu_avg)
        
        # Save the raw data for further analysis
        self._save_raw_data()
        
    def _plot_comparison(self, cpu_avg, gpu_avg):
        """Create a bar chart comparing CPU and GPU performance"""
        functions = list(cpu_avg.keys())
        cpu_times = [cpu_avg[f] for f in functions]
        gpu_times = [gpu_avg[f] for f in functions]
        
        x = np.arange(len(functions))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        cpu_bars = ax.bar(x - width/2, cpu_times, width, label='CPU', color='#3498db')
        gpu_bars = ax.bar(x + width/2, gpu_times, width, label='GPU', color='#2ecc71')
        
        ax.set_title('CPU vs GPU Performance Comparison', fontsize=16)
        ax.set_xlabel('Functions', fontsize=14)
        ax.set_ylabel('Average Time (ms)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(functions, fontsize=12)
        ax.legend(fontsize=12)
        
        # Add speedup text above bars
        for i, func in enumerate(functions):
            speedup = cpu_avg[func] / gpu_avg[func] if gpu_avg[func] > 0 else 0
            y_pos = max(cpu_times[i], gpu_times[i]) + 0.05 * max(cpu_times)
            ax.text(i, y_pos, 
                    f"{speedup:.2f}x",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
        
        # Add values on top of bars
        for i, v in enumerate(cpu_times):
            ax.text(i - width/2, v + 0.01 * max(cpu_times), 
                   f"{v:.1f}", ha='center', va='bottom', fontsize=10)
            
        for i, v in enumerate(gpu_times):
            ax.text(i + width/2, v + 0.01 * max(cpu_times), 
                   f"{v:.1f}", ha='center', va='bottom', fontsize=10)
        
        fig.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300)
        plt.close()
        print("Benchmark chart saved as 'benchmark_results.png'")
    
    def _save_raw_data(self):
        """Save the raw timing data for further analysis"""
        import json
        
        data = {
            "cpu": {k: [float(x) * 1000 for x in v] for k, v in self.cpu_times.items()},
            "gpu": {k: [float(x) * 1000 for x in v] for k, v in self.gpu_times.items()}
        }
        
        with open('benchmark_raw_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print("Raw benchmark data saved as 'benchmark_raw_data.json'")
    
    def compare_specific_frame(self, frame, frame_index=0):
        """Compare CPU and GPU on one specific frame with visualizations
        
        Args:
            frame: The frame to compare
            frame_index: Optional index for naming output files
        """
        print(f"\nDetailed comparison on frame {frame_index}...")
        
        # CPU run
        cpu_start = time.time()
        cpu_mask = self.detector_cpu.mask_on_white(frame.copy())
        cpu_bboxes = self.detector_cpu.find_ball(cpu_mask, frame.copy())
        cpu_result = frame.copy()
        if cpu_bboxes:
            for bbox in cpu_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(cpu_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cpu_time = time.time() - cpu_start
        
        # GPU run
        gpu_start = time.time()
        gpu_mask = self.detector_gpu.mask_on_white(frame.copy())
        gpu_bboxes = self.detector_gpu.find_ball(gpu_mask, frame.copy())
        gpu_result = frame.copy()
        if gpu_bboxes:
            for bbox in gpu_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(gpu_result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        gpu_time = time.time() - gpu_start
        
        # Create side-by-side visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame', fontsize=14)
        axes[0, 0].axis('off')
        
        # CPU mask
        axes[0, 1].imshow(cpu_mask, cmap='gray')
        axes[0, 1].set_title(f'CPU Mask ({cpu_time*1000:.2f} ms)', fontsize=14)
        axes[0, 1].axis('off')
        
        # CPU result
        axes[0, 2].imshow(cv2.cvtColor(cpu_result, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'CPU Result - {len(cpu_bboxes)} boxes', fontsize=14)
        axes[0, 2].axis('off')
        
        # Empty for alignment
        axes[1, 0].axis('off')
        
        # GPU mask
        axes[1, 1].imshow(gpu_mask, cmap='gray')
        axes[1, 1].set_title(f'GPU Mask ({gpu_time*1000:.2f} ms)', fontsize=14)
        axes[1, 1].axis('off')
        
        # GPU result
        axes[1, 2].imshow(cv2.cvtColor(gpu_result, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'GPU Result - {len(gpu_bboxes)} boxes', fontsize=14)
        axes[1, 2].axis('off')
        
        # Add speedup information
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        fig.suptitle(f'GPU vs CPU Comparison - {speedup:.2f}x Speedup', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'comparison_frame_{frame_index}.png', dpi=300)
        plt.close()
        print(f"Detailed comparison saved as 'comparison_frame_{frame_index}.png'")
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'cpu_bboxes': len(cpu_bboxes),
            'gpu_bboxes': len(gpu_bboxes),
            'speedup': speedup
        }


def prepare_files():
    """Save the original CPU implementation as ball_detect_cpu.py"""
    # Read the original file content and save as CPU implementation
    if not os.path.exists('ball_detect_cpu.py'):
        if os.path.exists('paste.txt'):
            with open('paste.txt', 'r') as f:
                content = f.read()
            
            with open('ball_detect_cpu.py', 'w') as f:
                f.write(content)
            
            print("Original implementation saved as ball_detect_cpu.py")
        else:
            print("Warning: paste.txt not found. Make sure ball_detect_cpu.py exists.")


def load_frames_from_folder(folder_path, limit=None):
    """Load frames from a folder of images
    
    Args:
        folder_path: Path to folder containing image frames
        limit: Optional maximum number of frames to load
        
    Returns:
        List of loaded frames as numpy arrays
    """
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg"))) + \
                 sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if limit is not None:
        image_files = image_files[:limit]
    
    frames = []
    for img_file in image_files:
        frame = cv2.imread(img_file)
        if frame is not None:
            frames.append(frame)
            
    print(f"Loaded {len(frames)} frames from {folder_path}")
    return frames


def load_frames_from_video(video_path, limit=None):
    """Load frames from a video file
    
    Args:
        video_path: Path to video file
        limit: Optional maximum number of frames to load
        
    Returns:
        List of loaded frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret or (limit is not None and len(frames) >= limit):
            break
        frames.append(frame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


# if __name__ == "__main__":
#     # First, ensure we have both implementations saved
#     prepare_files()
    
#     # Ask user for frames source
#     print("\nSelect frames source:")
#     print("1. Load from a folder of images")
#     print("2. Load from a video file")
#     print("3. I already have frames in a variable")
    
#     choice = input("Enter your choice (1-3): ").strip()
    
#     frames = []
    
#     if choice == '1':
#         folder_path = input("Enter path to folder with frames: ").strip()
#         limit = input("Maximum number of frames to process (leave empty for all): ").strip()
#         limit = int(limit) if limit else None
#         frames = load_frames_from_folder(folder_path, limit)
        
#     elif choice == '2':
#         video_path = input("Enter path to video file: ").strip()
#         limit = input("Maximum number of frames to process (leave empty for all): ").strip()
#         limit = int(limit) if limit else None
#         frames = load_frames_from_video(video_path, limit)
        
#     elif choice == '3':
#         print("\nTo use this option, you need to modify the script.")
#         print("Replace the code at the bottom with:")
#         print("    benchmark = BenchmarkTool()")
#         print("    # Replace 'your_frames' with your actual frames variable")
#         print("    benchmark.run_benchmark_on_frames(your_frames)")
#         print("\nAlternatively, you can import this module and call the functions directly.")
#         sys.exit(0)
    
#     if not frames:
#         print("No frames loaded. Exiting.")
#         sys.exit(1)
    

    
    # # Ask if user wants to see detailed comparison for specific frames
    # view_detailed = input("\nView detailed comparison for specific frames? (y/n): ").lower().strip()
    # if view_detailed == 'y':
    #     num_frames = min(5, len(frames))  # Default to 5 frames max
    #     custom_num = input(f"How many frames to compare in detail? (default: {num_frames}): ").strip()
    #     if custom_num:
    #         num_frames = min(int(custom_num), len(frames))
        
    #     # Select frames at regular intervals
    #     interval = max(1, len(frames) // num_frames)
    #     for i in range(0, len(frames), interval):
    #         if i // interval >= num_frames:
    #             break
    #         benchmark.compare_specific_frame(frames[i], i)