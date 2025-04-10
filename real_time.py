import multiprocessing
import time
import numpy as np
from mss import mss
import cv2
import keyboard  # For global key detection


def frame_processor(queue, stop_event):
    """
    Processes frames from the queue.

    :param queue: Multiprocessing queue to receive frames.
    :param stop_event: Event to stop processing.
    """
    while not stop_event.is_set():
        if not queue.empty():
            # Get a frame from the queue
            frame = queue.get()

            # Perform your processing here
            # Example: Display the frame
            # cv2.imshow("Processed Frame", frame)
            # print('frame_processed')

            # Example: Save the frame
            # cv2.imwrite(f"frame_{int(time.time())}.png", frame)

    # print("Frame processing stopped.")
    cv2.destroyAllWindows()


def screen_recorder(queue, stop_event, capture_event):
    """
    Captures the screen and sends frames to the queue when triggered by an event.

    :param queue: Multiprocessing queue to send frames.
    :param stop_event: Event to stop the recording.
    :param capture_event: Event to trigger a screenshot.
    """
    # Define the screen region to capture (entire screen in this case)
    monitor = {
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080,
    }  # Adjust to your screen resolution

    # Initialize mss
    with mss() as sct:
        while not stop_event.is_set():
            # Wait for the capture event to be set
            capture_event.wait()

            # Capture the screen
            frame = np.array(sct.grab(monitor))

            # Convert the frame from BGRA to BGR (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Put the frame in the queue
            if not queue.full():
                queue.put(frame)

            # Clear the capture event for the next trigger
            capture_event.clear()

    # print("Screen recording stopped.")


def global_key_listener(stop_event, capture_event):
    """
    Listens for global key presses to stop the processes or trigger a screenshot.

    :param stop_event: Event to signal stopping.
    :param capture_event: Event to trigger a screenshot.
    """
    # print("Press 's' to take a screenshot, 'q' to stop the processes.")
    while not stop_event.is_set():
        if keyboard.is_pressed("s"):  # Trigger a screenshot
            capture_event.set()
            # print("Screenshot triggered.")
            time.sleep(0.5)  # Debounce to avoid multiple triggers
        elif keyboard.is_pressed("q"):  # Stop the processes
            stop_event.set()
            # print("'q' pressed. Stopping processes...")
            break
        time.sleep(0.1)  # Small delay to avoid busy-waiting


if __name__ == "__main__":
    # Create a queue to share frames between processes
    queue = multiprocessing.Queue(maxsize=10)  # Adjust maxsize as needed

    # Create events to signal stopping and capturing
    stop_event = multiprocessing.Event()
    capture_event = multiprocessing.Event()

    # Create and start the screen recorder process
    recorder_process = multiprocessing.Process(
        target=screen_recorder, args=(queue, stop_event, capture_event)
    )
    recorder_process.start()

    # Create and start the frame processor process
    processor_process = multiprocessing.Process(
        target=frame_processor, args=(queue, stop_event)
    )
    processor_process.start()

    # Start the global key listener in the main process
    global_key_listener(stop_event, capture_event)

    # Wait for the processes to finish
    recorder_process.join()
    processor_process.join()

    # print("Main program exited.")
