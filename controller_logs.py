import time
import numpy as np
import cv2
import keyboard
import os
import logging
from datetime import datetime
from mss import mss
from ocr_extractor import OCRExtractor
from computer_vision import ComputerVision
from rl import DQNAgent, get_state_from_positions, calculate_reward


class Controller:
    """
    Main controller class for an AI-powered soccer game automation system.
    This class integrates computer vision, OCR, and reinforcement learning to
    autonomously play a soccer game.
    """

    def __init__(self, model_path=None, record_video=False):
        # Existing initialization code...

        # Set up logging
        self.setup_logging()
        self.logger.info("Initializing Controller")

        # Video recording setting
        self.record_video = record_video
        self.video_writer = None
        self.fps = 30.0  # Target FPS for video recording
        self.frame_size = (2560, 1600)  # Default frame size

        # Initialize OCR and computer vision modules
        self.ocr_extractor = OCRExtractor()
        self.computer_vision = ComputerVision()
        self.logger.info("OCR and Computer Vision modules initialized")

        # Frame tracking variables - IMPROVED COUNTERS
        self.frame_count = 0  # Total frames captured
        self.processed_frames = 0  # Frames actually processed
        self.skipped_frames = 0  # Frames intentionally skipped
        self.buffered_frames = 0  # Frames currently in buffer

        # Frame skipping parameters for game control
        self.pause_frames_skip = 1  # Number of frames to skip after pausing
        self.between_attacks_skip = 0  # Frames to skip between game attacks
        self.first_start_skip = 14  # Frames to skip at initial game start
        self.skip_counter = 14  # Counter for skipping frames

        # Game state tracking
        self.last_attempt = None  # Track the last attempt number
        self.last_points = None  # Track the last score points
        self.temp_flag = False  # Flag for temporary state control
        self.frames_to_send = []  # Buffer for frames to process
        self.first_time = True  # Flag for first action

        # Timing tracking for FPS calculation
        self.start_time = time.time()
        self.frame_times = []
        self.current_fps = 0

        # Reinforcement learning setup
        self.logger.info(
            f"Initializing DQN Agent with model path: {model_path if model_path else 'None - creating new model'}"
        )
        self.rl_agent = DQNAgent(load_path=model_path) if model_path else DQNAgent()
        self.trainNN_counter = 0  # Counter for neural network training
        self.action_taken = None  # Current action
        self.prev_action = None  # Previous action
        self.last_action_time = None  # Time when last action was taken

        # Default scene state when data is unavailable
        self.dummy_scene = {
            "Attempts": None,
            "Score": None,
        }

        self.logger.info("Controller initialization complete")
        if self.record_video:
            self.logger.info("Video recording is enabled")

    def _get_action_description(self, action):
        """
        Get a human-readable description of an action.

        Args:
            action (int): The action ID

        Returns:
            str: Description of the action
        """
        action_descriptions = {
            0: "Pass North",
            1: "Pass Northeast",
            2: "Pass East",
            3: "Pass Southeast",
            4: "Pass South",
            5: "Pass Southwest",
            6: "Pass West",
            7: "Pass Northwest",
            8: "Dribble North",
            9: "Dribble Northeast",
            10: "Dribble East",
            11: "Dribble Southeast",
            12: "Dribble South",
            13: "Dribble Southwest",
            14: "Dribble West",
            15: "Dribble Northwest",
            16: "Shoot",
        }

        return action_descriptions.get(action, f"Unknown action: {action}")

    def setup_logging(self):
        """
        Set up the logging configuration with timestamps and file output.
        """
        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Create a unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/controller_log_{timestamp}.log"

        # Configure logger
        self.logger = logging.getLogger("SoccerController")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to handlers
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging initialized: {log_filename}")

    def setup_video_recording(self, output_folder, resolution=(2560, 1600)):
        """
        Set up video recording with logs overlay.

        Args:
            output_folder (str): Folder to save the video file
            resolution (tuple): Resolution of the video (width, height)
        """
        if not self.record_video:
            return

        # Store output folder for saving individual frames
        self.output_folder = output_folder

        # Create annotated frames directory
        annotated_frames_dir = f"{output_folder}/annotated_frames"
        if not os.path.exists(annotated_frames_dir):
            os.makedirs(annotated_frames_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{output_folder}/gameplay_recording_{timestamp}.mp4"

        # Initialize the video writer using mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            video_filename, fourcc, self.fps, resolution
        )

        self.frame_size = resolution
        self.logger.info(f"Video recording initialized: {video_filename}")
        self.logger.info(f"Annotated frames will be saved to: {annotated_frames_dir}/")

    def close_video_recording(self):
        """
        Properly close video recording if enabled.
        """
        if self.record_video and self.video_writer is not None:
            self.video_writer.release()
            self.logger.info("Video recording completed and saved")

    def add_logs_to_frame(
        self, frame, frame_count, ocr_results=None, scene=None, has_ball=None
    ):
        """
        Add enhanced real-time logging information to a frame.

        Args:
            frame (np.array): The frame to add logs to
            frame_count (int): Current frame number
            ocr_results (dict, optional): Results from OCR
            scene (dict, optional): Scene analysis results
            has_ball (bool, optional): Whether ball possession is detected

        Returns:
            np.array: Frame with logging information overlay
        """
        # Create a copy of the frame
        annotated_frame = frame.copy()

        # Calculate current FPS
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Update FPS calculation (rolling average of last 30 frames)
        self.frame_times.append(current_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            if len(self.frame_times) >= 2:
                self.current_fps = len(self.frame_times) / (
                    self.frame_times[-1] - self.frame_times[0]
                )

        # Define text color and font parameters
        color = (0, 255, 255)  # Yellow
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 30

        # Create a semi-transparent overlay for the logging area
        overlay_height = 200  # Increased height for more log information

        # Create a separate overlay
        overlay = np.zeros_like(annotated_frame[:overlay_height])
        cv2.rectangle(
            overlay, (0, 0), (self.frame_size[0], overlay_height), (0, 0, 0), -1
        )

        # Blend the overlay with the top portion of the frame
        cv2.addWeighted(
            overlay,
            0.7,
            annotated_frame[:overlay_height],
            0.3,
            0,
            annotated_frame[:overlay_height],
        )

        # Add basic information (frame count, time, FPS)
        lines = [
            f"Frame: {frame_count} | Time: {elapsed:.1f}s | FPS: {self.current_fps:.1f}",
            f"Attempts: {self.last_attempt if self.last_attempt is not None else 'N/A'} | Score: {self.last_points if self.last_points is not None else 'N/A'}",
            f"Game State: {'PAUSED' if self.temp_flag else 'RUNNING'} | Skip Counter: {self.skip_counter}",
        ]

        # Add OCR results if available
        if ocr_results:
            ocr_line = f"OCR: Attempts={ocr_results.get('attempts', 'None')} | Points={ocr_results.get('points', 'None')}"
            lines.append(ocr_line)

        # Add action information with more details
        action_str = "None"
        if self.action_taken is not None:
            action_str = self._get_action_description(self.action_taken)
        elif self.prev_action is not None:
            action_str = f"{self._get_action_description(self.prev_action)} (previous)"

        lines.append(f"Action: {action_str}")

        # Add ball possession information
        if has_ball is not None:
            lines.append(f"Ball Possession: {'Yes' if has_ball else 'No'}")

        # Add buffer information
        lines.append(
            f"Buffer: {len(self.frames_to_send)}/5 frames | Total Frames: {self.processed_frames + self.skipped_frames}"
        )

        # Add frame processing info
        lines.append(
            f"Processed: {self.processed_frames} | Skipped: {self.skipped_frames} | Buffered: {self.buffered_frames}"
        )

        # Draw all lines of text
        for i, line in enumerate(lines):
            y_position = (i + 1) * line_height
            cv2.putText(
                annotated_frame,
                line,
                (10, y_position),
                font,
                font_scale,
                color,
                thickness,
            )

        return annotated_frame

    def write_frame_to_video(
        self, frame, frame_count, ocr_results=None, scene=None, has_ball=None
    ):
        """
        Add logs to frame and write it to video. Records ALL frames regardless of state.

        Args:
            frame (np.array): Frame to write
            frame_count (int): Current frame count
            ocr_results (dict, optional): OCR extraction results
            scene (dict, optional): Scene analysis results
            has_ball (bool, optional): Ball possession status
        """
        if not self.record_video or self.video_writer is None:
            return

        # Add logs to the frame
        annotated_frame = self.add_logs_to_frame(
            frame, frame_count, ocr_results, scene, has_ball
        )

        # Save the annotated frame as an image for reference
        timestamp = datetime.now().strftime("%H%M%S_%f")
        cv2.imwrite(
            f"{self.output_folder}/annotated_frame_{frame_count:06d}_{timestamp}.jpg",
            annotated_frame,
        )

        # Write the frame to video
        self.video_writer.write(annotated_frame)

        self.logger.debug(f"Frame {frame_count} recorded to video with annotations")

    def pause_game(self):
        """
        Pause the game by simulating ESC key press.
        """
        self.logger.info("Pausing game")
        keyboard.press("esc")
        time.sleep(0.1)
        keyboard.release("esc")

    def resume_game(self):
        """
        Resume the game by simulating ENTER key press and reset relevant counters.
        """
        self.logger.info("Resuming game")
        keyboard.press("enter")
        time.sleep(0.1)
        keyboard.release("enter")
        self.skip_counter = self.pause_frames_skip
        self.temp_flag = False
        self.logger.debug(f"Reset skip_counter to {self.pause_frames_skip}")

    def make_action(
        self, action, pass_duration=0.5, dribble_duration=0.5, shoot_duration=0.2
    ):
        """
        Execute the specified game action.
        This method is to be implemented based on the specific game controls.

        Args:
            action: The action to take in the game environment
        """
        action_descriptions = {
            0: "Passing the ball to the north",
            1: "Passing the ball to the northeast",
            2: "Passing the ball to the east",
            3: "Passing the ball to the southeast",
            4: "Passing the ball to the south",
            5: "Passing the ball to the southwest",
            6: "Passing the ball to the west",
            7: "Passing the ball to the northwest",
            8: "Dribbling the ball to the north",
            9: "Dribbling the ball to the northeast",
            10: "Dribbling the ball to the east",
            11: "Dribbling the ball to the southeast",
            12: "Dribbling the ball to the south",
            13: "Dribbling the ball to the southwest",
            14: "Dribbling the ball to the west",
            15: "Dribbling the ball to the northwest",
            16: "Shooting the ball",
        }

        action_description = action_descriptions.get(
            action, f"Unknown action: {action}"
        )
        self.logger.info(f"Making action [{action}]: {action_description}")
        self.last_action_time = time.time()

        match action:
            case 0:
                keyboard.press("w")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("w")
            case 1:
                keyboard.press("w")
                keyboard.press("d")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("d")
                keyboard.release("w")
            case 2:
                keyboard.press("d")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("d")
            case 3:
                keyboard.press("s")
                keyboard.press("d")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("d")
                keyboard.release("s")
            case 4:
                keyboard.press("s")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("s")
            case 5:
                keyboard.press("s")
                keyboard.press("a")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("a")
                keyboard.release("s")
            case 6:
                keyboard.press("a")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("a")
            case 7:
                keyboard.press("w")
                keyboard.press("a")
                keyboard.press("p")
                time.sleep(pass_duration)
                keyboard.release("p")
                keyboard.release("a")
                keyboard.release("w")
            case 8:
                keyboard.press("w")
                keyboard.press("d")
                time.sleep(dribble_duration)
                keyboard.release("d")
                keyboard.release("w")
            case 9:
                keyboard.press("w")
                keyboard.press("d")
                time.sleep(dribble_duration)
                keyboard.release("d")
                keyboard.release("w")
            case 10:
                keyboard.press("d")
                time.sleep(dribble_duration)
                keyboard.release("d")
            case 11:
                keyboard.press("s")
                keyboard.press("d")
                time.sleep(dribble_duration)
                keyboard.release("d")
                keyboard.release("s")
            case 12:
                keyboard.press("s")
                time.sleep(dribble_duration)
                keyboard.release("s")
            case 13:
                keyboard.press("s")
                keyboard.press("a")
                time.sleep(dribble_duration)
                keyboard.release("a")
                keyboard.release("s")
            case 14:
                keyboard.press("a")
                time.sleep(dribble_duration)
                keyboard.release("a")
            case 15:
                keyboard.press("w")
                keyboard.press("a")
                time.sleep(dribble_duration)
                keyboard.release("a")
                keyboard.release("w")
            case 16:
                keyboard.press("d")
                keyboard.press("k")
                time.sleep(shoot_duration)
                keyboard.release("k")
                keyboard.release("d")
            case _:
                self.logger.warning(f"Unknown action: {action}")

        self.logger.debug(f"Action {action} completed")

    def run(self):
        """
        Main execution loop for the controller with improved video logging.
        Captures ALL screen frames, processes them with OCR and computer vision,
        and uses reinforcement learning to make gameplay decisions.
        """
        # Set up screen capture parameters for a wide screen monitor
        monitor = {"top": 0, "left": 0, "width": 2560, "height": 1600}

        # Create directory for saving captured frames
        output_folder = "captured_frames"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"{output_folder}_{timestamp}"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            self.logger.info(f"Created output directory: {output_folder}")

        # Set up video recording if enabled
        if self.record_video:
            self.setup_video_recording(
                output_folder, resolution=(monitor["width"], monitor["height"])
            )

        # Create directory for saving ALL frames
        all_frames_dir = f"{output_folder}/all_frames"
        if not os.path.exists(all_frames_dir):
            os.makedirs(all_frames_dir)
            self.logger.info(f"Created all frames directory: {all_frames_dir}")

        with mss() as sct:
            self.logger.info("Press 's' to start capturing, 'q' to quit")

            # Wait for start command
            while not keyboard.is_pressed("s"):
                time.sleep(0.1)
                if keyboard.is_pressed("q"):
                    self.logger.info("Quit command detected before starting")
                    return

            # Start the game
            keyboard.press("enter")
            time.sleep(0.1)
            keyboard.release("enter")
            self.logger.info("Starting capture cycle...")

            running = True
            self.temp_flag = False
            self.start_time = time.time()

            # Reset frame counters
            self.frame_count = 0
            self.processed_frames = 0
            self.skipped_frames = 0
            self.buffered_frames = 0

            # Frame sequence tracking for accurate frame numbering
            frame_sequence = 0

            # Game state tracking
            paused = False
            action_log = "No action yet"

            # Main capture and processing loop
            while running:
                current_time = time.time()
                elapsed = current_time - self.start_time

                # Capture current screen frame
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Increment frame counter consistently for every captured frame
                self.frame_count += 1
                frame_sequence += (
                    1  # Additional sequence counter for accurate file naming
                )

                # Save EVERY frame regardless of state
                all_frame_filename = (
                    f"{all_frames_dir}/frame_all_{frame_sequence:06d}.jpg"
                )
                cv2.imwrite(all_frame_filename, frame)

                # Extract OCR results for logging
                ocr_results = self.ocr_extractor.extract_points_and_attempts(frame)

                # Always record every frame with current information
                self.write_frame_to_video(
                    frame,
                    self.frame_count,
                    ocr_results,
                    scene=None if len(self.frames_to_send) == 0 else {},
                    has_ball=None,
                )

                # Process frames based on current state
                if self.skip_counter > 0:
                    # Even though we're skipping processing, we still save the frame
                    self.skip_counter -= 1
                    self.skipped_frames += 1
                    self.logger.debug(
                        f"Skipping processing for frame {self.frame_count}, skip_counter: {self.skip_counter}"
                    )

                    # Track pause state for logging
                    if not paused:
                        paused_frame_filename = (
                            f"{output_folder}/paused_frame_{frame_sequence:06d}.jpg"
                        )
                        cv2.imwrite(paused_frame_filename, frame)
                        self.logger.info(
                            f"Game is being skipped at frame {self.frame_count}"
                        )

                    continue

                # Handle temporary state flag for frame processing control
                if self.temp_flag == False:
                    # Log frame capture and save with accurate sequential numbering
                    frame_filename = f"{output_folder}/frame_{frame_sequence:06d}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    self.logger.debug(
                        f"Saved frame {self.frame_count} (seq: {frame_sequence}) to {frame_filename}"
                    )

                    self.temp_flag = True
                    self.processed_frames += 1
                    self.frames_to_send.clear()
                    self.buffered_frames = 0  # Reset buffer counter
                    self.logger.debug("Cleared frames_to_send buffer")
                    continue

                # Process accumulated frames when buffer is full
                if len(self.frames_to_send) == 5:
                    self.logger.info(
                        f"Processing buffer of {len(self.frames_to_send)} frames at elapsed time: {elapsed:.2f}s"
                    )

                    # Mark that we're pausing - log this in frames
                    paused = True
                    pause_start_time = time.time()
                    pause_frame = frame.copy()

                    # Save special pause indicator frame
                    pause_filename = (
                        f"{output_folder}/pause_start_{frame_sequence:06d}.jpg"
                    )
                    cv2.putText(
                        pause_frame,
                        "GAME PAUSING",
                        (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 255),
                        4,
                    )
                    cv2.imwrite(pause_filename, pause_frame)

                    # Pause game to process frames
                    self.pause_game()

                    # Save frames for analysis with accurate sequence numbering
                    for i, buffer_frame in enumerate(self.frames_to_send):
                        buffer_frame_filename = f"{output_folder}/frame_buffer_{frame_sequence-len(self.frames_to_send)+i+1:06d}.jpg"
                        cv2.imwrite(buffer_frame_filename, buffer_frame)
                        self.logger.debug(
                            f"Saved buffered frame {i+1}/5 to {buffer_frame_filename}"
                        )

                    # Get scene information from computer vision module
                    self.logger.info(
                        "Running computer vision analysis on buffered frames"
                    )
                    before_time = time.time()
                    scene, has_ball = self.computer_vision.run(
                        self.frames_to_send, self.last_attempt
                    )
                    cv_time = time.time() - before_time
                    self.logger.info(f"COMP_VIS_TIME {cv_time:.2f}s")
                    self.logger.info(f"Scene analysis: {scene}")
                    self.logger.info(f"Ball possession: {has_ball}")

                    # Capture additional frames during pause for video recording
                    pause_analysis_frame = np.array(sct.grab(monitor))
                    pause_analysis_frame = cv2.cvtColor(
                        pause_analysis_frame, cv2.COLOR_BGRA2BGR
                    )

                    self.frame_count += 1
                    frame_sequence += 1

                    # Add analysis information to the frame
                    pause_analysis_filename = (
                        f"{output_folder}/pause_analysis_{frame_sequence:06d}.jpg"
                    )
                    analysis_text = f"CV Time: {cv_time:.2f}s | Ball: {has_ball}"
                    cv2.putText(
                        pause_analysis_frame,
                        analysis_text,
                        (10, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imwrite(pause_analysis_filename, pause_analysis_frame)

                    # Record the analysis frame
                    self.write_frame_to_video(
                        pause_analysis_frame,
                        self.frame_count,
                        ocr_results,
                        scene,
                        has_ball,
                    )

                    # Clear buffer and reset counter
                    self.frames_to_send.clear()
                    self.buffered_frames = 0

                    # Compute RL time taken
                    self.logger.info("Running reinforcement learning decision making")
                    before_time = time.time()
                    action_taken = None

                    if has_ball:
                        # Reinforcement learning decision making
                        scene_toRL = scene.copy()
                        scene_toRL["Attempts"] = self.last_attempt
                        scene_toRL["Score"] = self.last_points
                        cur_state = get_state_from_positions(scene_toRL)
                        prev_state = self.rl_agent.prev_state

                        # Handle first action or subsequent actions
                        if self.first_time:
                            self.first_time = False
                            action_taken = self.rl_agent.act(state=cur_state)
                            self.logger.info(f"First action taken: {action_taken}")
                            action_log = f"First action: {self._get_action_description(action_taken)}"
                            self.rl_agent.prev_state = cur_state
                        else:
                            action_taken = self.rl_agent.act(state=cur_state)
                            self.logger.info(f"Action taken: {action_taken}")
                            action_log = (
                                f"Action: {self._get_action_description(action_taken)}"
                            )
                            reward = calculate_reward(
                                prev_state, cur_state, self.prev_action
                            )
                            self.logger.info(f"Calculated reward: {reward}")
                            action_log += f" | Reward: {reward:.2f}"
                            self.rl_agent.remember(
                                prev_state,
                                self.prev_action,
                                reward,
                                cur_state,
                                done=False,
                            )

                        # Periodically train the neural network
                        self.trainNN_counter += 1
                        if self.trainNN_counter == 10:
                            self.logger.info(
                                "Training neural network on batch of experiences"
                            )
                            self.rl_agent.replay(32)  # Train on batch of 32 experiences
                            self.trainNN_counter = 0

                        # Store actions for next iteration
                        self.logger.info(f"Setting action to take: {action_taken}")
                        self.action_taken = action_taken
                        self.prev_action = action_taken
                    else:
                        self.logger.info("No ball possession detected, no action taken")
                        action_log = "No ball possession - no action"

                    rl_time = time.time() - before_time
                    self.logger.info(f"RL_TIME {rl_time:.2f}s")

                    # Capture another frame for decision visualization
                    decision_frame = np.array(sct.grab(monitor))
                    decision_frame = cv2.cvtColor(decision_frame, cv2.COLOR_BGRA2BGR)

                    self.frame_count += 1
                    frame_sequence += 1

                    # Add decision information to the frame
                    decision_filename = (
                        f"{output_folder}/decision_{frame_sequence:06d}.jpg"
                    )
                    decision_text = f"RL Time: {rl_time:.2f}s | Action: {action_log}"
                    cv2.putText(
                        decision_frame,
                        decision_text,
                        (10, frame.shape[0] // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imwrite(decision_filename, decision_frame)

                    # Record the decision frame
                    self.write_frame_to_video(
                        decision_frame, self.frame_count, ocr_results, scene, has_ball
                    )

                    # Capture unpausing
                    unpause_frame = np.array(sct.grab(monitor))
                    unpause_frame = cv2.cvtColor(unpause_frame, cv2.COLOR_BGRA2BGR)

                    self.frame_count += 1
                    frame_sequence += 1

                    # Add unpausing information
                    unpause_filename = (
                        f"{output_folder}/unpause_{frame_sequence:06d}.jpg"
                    )
                    pause_duration = time.time() - pause_start_time
                    unpause_text = (
                        f"GAME RESUMING | Pause duration: {pause_duration:.2f}s"
                    )
                    cv2.putText(
                        unpause_frame,
                        unpause_text,
                        (frame.shape[1] // 2 - 300, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                    )
                    cv2.imwrite(unpause_filename, unpause_frame)

                    # Record the unpause frame
                    self.write_frame_to_video(
                        unpause_frame, self.frame_count, ocr_results, scene, has_ball
                    )

                    # Resume game and continue
                    self.resume_game()
                    paused = False

                    if self.action_taken is not None:
                        # Capture pre-action frame
                        pre_action_frame = np.array(sct.grab(monitor))
                        pre_action_frame = cv2.cvtColor(
                            pre_action_frame, cv2.COLOR_BGRA2BGR
                        )

                        self.frame_count += 1
                        frame_sequence += 1

                        pre_action_filename = (
                            f"{output_folder}/pre_action_{frame_sequence:06d}.jpg"
                        )
                        pre_action_text = f"ABOUT TO EXECUTE: {self._get_action_description(self.action_taken)}"
                        cv2.putText(
                            pre_action_frame,
                            pre_action_text,
                            (frame.shape[1] // 2 - 300, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 165, 0),
                            3,
                        )
                        cv2.imwrite(pre_action_filename, pre_action_frame)

                        # Record the pre-action frame
                        self.write_frame_to_video(
                            pre_action_frame,
                            self.frame_count,
                            ocr_results,
                            scene,
                            has_ball,
                        )

                        # Execute the action
                        self.make_action(self.action_taken)
                        self.logger.info(f"Action executed: {self.action_taken}")
                        action_log = f"Executed: {self._get_action_description(self.action_taken)}"
                        self.action_taken = None

                    self.processed_frames += 1

                    continue

                # Add frame to buffer
                self.frames_to_send.append(frame)
                self.buffered_frames += 1
                self.logger.debug(
                    f"Added frame to buffer, current size: {len(self.frames_to_send)}/5, frame count: {self.frame_count}, sequence: {frame_sequence}"
                )

                # Handle game restart condition
                if self.last_attempt == "0" and ocr_results["attempts"] is None:
                    self.action_taken = None
                    self.logger.info(
                        f"No attempts detected at frame {self.frame_count}, attempting restart"
                    )

                    # Capture restart frame
                    restart_frame = frame.copy()
                    restart_filename = (
                        f"{output_folder}/game_restart_{frame_sequence:06d}.jpg"
                    )
                    restart_text = "GAME RESTARTING"
                    cv2.putText(
                        restart_frame,
                        restart_text,
                        (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 255),
                        4,
                    )
                    cv2.imwrite(restart_filename, restart_frame)

                    # Record the restart frame
                    self.write_frame_to_video(
                        restart_frame, self.frame_count, ocr_results
                    )

                    time.sleep(3)
                    keyboard.press("enter")
                    time.sleep(0.1)
                    keyboard.release("enter")
                    self.skip_counter = self.pause_frames_skip

                    # Process end of game episode
                    scene_toRL = self.dummy_scene.copy()
                    scene_toRL["Attempts"] = self.last_attempt
                    scene_toRL["Score"] = self.last_points
                    cur_state = get_state_from_positions(scene_toRL)
                    prev_state = self.rl_agent.prev_state
                    self.last_attempt = None
                    self.temp_flag = False
                    self.processed_frames += 1

                    # Store terminal state transition for learning
                    reward = calculate_reward(prev_state, cur_state, self.prev_action)
                    self.logger.info(f"End of episode reward: {reward}")
                    action_log = f"End of episode | Reward: {reward:.2f}"
                    self.rl_agent.remember(
                        prev_state, self.prev_action, reward, cur_state, done=True
                    )
                    self.first_time = True
                    continue

                # Track attempts and handle attempt transitions
                if self.last_attempt is None:
                    self.last_attempt = ocr_results["attempts"]
                    self.logger.info(f"Initial attempt detected: {self.last_attempt}")
                else:
                    if ocr_results["attempts"] != self.last_attempt:
                        self.action_taken = None
                        # New attempt detected - process end of previous attempt
                        self.logger.info(
                            f"New attempt detected: {ocr_results['attempts']} (was {self.last_attempt})"
                        )

                        # Capture attempt transition frame
                        attempt_frame = frame.copy()
                        attempt_filename = (
                            f"{output_folder}/new_attempt_{frame_sequence:06d}.jpg"
                        )
                        attempt_text = f"NEW ATTEMPT: {ocr_results['attempts']} (was {self.last_attempt})"
                        cv2.putText(
                            attempt_frame,
                            attempt_text,
                            (frame.shape[1] // 2 - 300, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 0),
                            3,
                        )
                        cv2.imwrite(attempt_filename, attempt_frame)

                        # Record the attempt transition frame
                        self.write_frame_to_video(
                            attempt_frame, self.frame_count, ocr_results
                        )

                        self.last_attempt = ocr_results["attempts"]
                        scene_toRL = self.dummy_scene.copy()
                        scene_toRL["Attempts"] = self.last_attempt
                        scene_toRL["Score"] = self.last_points

                        # Store transition with terminal state
                        cur_state = get_state_from_positions(scene_toRL)
                        prev_state = self.rl_agent.prev_state
                        reward = calculate_reward(
                            prev_state, cur_state, self.prev_action
                        )
                        self.logger.info(f"End of attempt reward: {reward}")
                        action_log = f"New attempt | Reward: {reward:.2f}"
                        self.rl_agent.remember(
                            prev_state, self.prev_action, reward, cur_state, done=True
                        )
                        self.first_time = True

                        self.logger.info(
                            f"Skipping {self.between_attacks_skip} frames between attempts"
                        )
                        self.skip_counter = self.between_attacks_skip
                        self.temp_flag = False
                        self.processed_frames += 1
                        self.frames_to_send.clear()
                        self.buffered_frames = 0
                        continue

                # Track score points
                if ocr_results["points"] is not None:
                    if self.last_points != ocr_results["points"]:
                        self.logger.info(
                            f"Score changed: {ocr_results['points']} (was {self.last_points})"
                        )

                        # Capture score change frame
                        score_frame = frame.copy()
                        score_filename = (
                            f"{output_folder}/score_change_{frame_sequence:06d}.jpg"
                        )
                        score_text = f"SCORE CHANGED: {ocr_results['points']} (was {self.last_points})"
                        cv2.putText(
                            score_frame,
                            score_text,
                            (frame.shape[1] // 2 - 300, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3,
                        )
                        cv2.imwrite(score_filename, score_frame)

                        # Record the score change frame
                        self.write_frame_to_video(
                            score_frame, self.frame_count, ocr_results
                        )

                        self.last_points = ocr_results["points"]
                        action_log = f"Score changed to {self.last_points}"

                # Check for quit command
                if keyboard.is_pressed("q"):
                    self.logger.info(
                        "Quit command detected at elapsed time: {:.2f}s".format(elapsed)
                    )
                    running = False
                    model_save_path = "soccer_dqn_trained.pth"
                    self.logger.info(f"Saving DQN model to {model_save_path}")
                    self.rl_agent.save(model_save_path)

                self.processed_frames += 1

            # Close video recording if enabled
            self.close_video_recording()

            # Print summary when finished
            total_time = time.time() - self.start_time
            self.logger.info("FIFA Vision has stopped.")
            self.logger.info("\n===== SUMMARY =====")
            self.logger.info(f"Total frames captured: {self.frame_count}")
            self.logger.info(f"Frames processed: {self.processed_frames}")
            self.logger.info(f"Frames skipped: {self.skipped_frames}")
            self.logger.info(f"Total runtime: {total_time:.2f} seconds")
            self.logger.info(
                f"Average capture FPS: {self.frame_count / total_time:.2f}"
            )
            self.logger.info(
                f"Average processing FPS: {self.processed_frames / total_time:.2f}"
            )
            self.logger.info(f"Frames saved to: {output_folder}/")
            self.logger.info(f"All frames saved to: {all_frames_dir}/")
            if self.record_video:
                self.logger.info(f"Video recording saved to: {output_folder}/")
            self.logger.info("==================")


if __name__ == "__main__":
    # Create and run the controller with optional video recording
    controller = Controller(
        model_path="./soccer_dqn_trained.pth",
        record_video=True,  # Set to False to disable video recording
    )
    controller.run()
