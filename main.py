import multiprocessing
import time
import numpy as np
from mss import mss
import cv2
import keyboard 
from ultralytics import YOLO
import os

def frame_processor(queue, stop_event, capture_event, response_queue):
    """
    Processes frames from the queue, waits 2 seconds, and sends a response back to the screen recorder.
    Saves the processed frames with YOLO predictions to a folder.
    
    :param queue: Multiprocessing queue to receive frames.
    :param stop_event: Event to stop processing.
    :param capture_event: Event to signal the screen recorder to capture the next frame.
    :param response_queue: Queue to send a response (string) back to the screen recorder.
    """

    model = YOLO('yolov8n.pt')  

    output_folder = "yolo_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0  

    while not stop_event.is_set():
        if not queue.empty():
            frame = queue.get()
            
            results = model.predict(frame, save=False)  

            annotated_frame = results[0].plot()  

            output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            print(f"Processed frame saved to {output_path}")

            frame_count += 1

            time.sleep(2)
            
            
            response_queue.put("Processing complete. Capture next frame.") # raga3 ay haga hena

            
            capture_event.set()

    print("Frame processing stopped.")

def screen_recorder(queue, stop_event, capture_event, response_queue):

    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust monitor dimensions as needed

    # Initialize mss
    with mss() as sct:
        while not stop_event.is_set():
            capture_event.wait()
            time.sleep(2)

            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if not queue.full():
                queue.put(frame)
                print("Frame captured and sent to processor.")

            if not response_queue.empty():
                response = response_queue.get()
                print(f"Response from processor: {response}")

            capture_event.clear()

    print("Screen recording stopped.")

def global_key_listener(stop_event, capture_event):

    print("Press 's' to start capturing, 'q' to stop the processes.")
    while not stop_event.is_set():
        if keyboard.is_pressed('s'):  # Start capturing
            capture_event.set()
            print("Capture triggered.")
            time.sleep(0.5)  # Debounce to avoid multiple triggers
        elif keyboard.is_pressed('q'):  # Stop the processes
            stop_event.set()
            print("'q' pressed. Stopping processes...")
            break
        time.sleep(0.1)  # Small delay to avoid busy-waiting

if __name__ == "__main__":
    # Create a queue to share frames between processes
    queue = multiprocessing.Queue(maxsize=10)  # Adjust maxsize as needed

    # Create a queue to send responses from the processor to the screen recorder
    response_queue = multiprocessing.Queue(maxsize=10)

    # Create events to signal stopping and capturing
    stop_event = multiprocessing.Event()
    capture_event = multiprocessing.Event()

    # Create and start the screen recorder process
    recorder_process = multiprocessing.Process(
        target=screen_recorder,
        args=(queue, stop_event, capture_event, response_queue)
    )
    recorder_process.start()

    # Create and start the frame processor process
    processor_process = multiprocessing.Process(
        target=frame_processor,
        args=(queue, stop_event, capture_event, response_queue)
    )
    processor_process.start()

    # Start the global key listener in the main process
    global_key_listener(stop_event, capture_event)

    # Wait for the processes to finish
    recorder_process.join()
    processor_process.join()

    print("Main program exited.")