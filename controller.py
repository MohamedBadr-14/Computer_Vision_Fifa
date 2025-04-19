import time
import numpy as np
import cv2
import keyboard
import os
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
    def __init__(self, model_path=None):
        """
        Initialize the controller with various components and parameters.
        
        Args:
            model_path (str, optional): Path to a pre-trained DQN model. If None, a new model is created.
        """
        # Initialize OCR and computer vision modules
        self.ocr_extractor = OCRExtractor()
        self.computer_vision = ComputerVision()
        
        # Frame skipping parameters for game control
        self.pause_frames_skip = 20     # Number of frames to skip after pausing
        self.between_attacks_skip = 10  # Frames to skip between game attacks
        self.first_start_skip = 110     # Frames to skip at initial game start
        self.skip_counter = 110         # Counter for skipping frames
        
        # Game state tracking
        self.last_attempt = None        # Track the last attempt number
        self.last_points = None         # Track the last score points
        self.temp_flag = False          # Flag for temporary state control
        self.frames_to_send = []        # Buffer for frames to process
        self.first_time = True          # Flag for first action
        
        # Reinforcement learning setup
        self.rl_agent = DQNAgent(load_path=model_path) if model_path else DQNAgent()
        self.trainNN_counter = 0        # Counter for neural network training
        self.action_taken = None        # Current action
        self.prev_action = None         # Previous action
        
        # Default scene state when data is unavailable
        self.dummy_scene = {
            "Attemps": None,
            "Score": None,
        }

    def pause_game(self):
        """
        Pause the game by simulating ESC key press.
        """
        keyboard.press('esc')
        time.sleep(0.1)
        keyboard.release('esc')

    def resume_game(self):
        """
        Resume the game by simulating ENTER key press and reset relevant counters.
        """
        keyboard.press('enter')
        time.sleep(0.1)
        keyboard.release('enter')
        self.skip_counter = self.pause_frames_skip
        self.temp_flag = False

    def take_action(self, action,pass_duration=0.5,dribble_duration=0.5,shoot_duration=0.2):
        """
        Execute the specified game action.
        This method is to be implemented based on the specific game controls.
        
        Args:
            action: The action to take in the game environment
        """
        match action:
            case 0:
                print("Passing the ball to the north")
                keyboard.press('w')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('w')
            case 1:
                print("Passing the ball to the northeast")
                keyboard.press('w')
                keyboard.press('d')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('d')
                keyboard.release('w')
            case 2:
                print("Passing the ball to the east")
                keyboard.press('d')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('d')
            case 3:
                print("Passing the ball to the southeast")
                keyboard.press('s')
                keyboard.press('d')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('d')
                keyboard.release('s')
            case 4:
                print("Passing the ball to the south")
                keyboard.press('s')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('s')
            case 5:
                print("Passing the ball to the southwest")
                keyboard.press('s')
                keyboard.press('a')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('a')
                keyboard.release('s')
            case 6:
                print("Passing the ball to the west")
                keyboard.press('a')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('a')
            case 7:
                print("Passing the ball to the northwest")
                keyboard.press('w')
                keyboard.press('a')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('a')
                keyboard.release('w')
            case 8:
                print("Dribbling the ball to the north")
                keyboard.press('w')
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
                keyboard.release('w')
            case 9:
                print("Dribbling the ball to the northeast")
                keyboard.press('w')
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
                keyboard.release('w')
            case 10:
                print("Dribbling the ball to the east")
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
            case 11:
                print("Dribbling the ball to the southeast")
                keyboard.press('s')
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
                keyboard.release('s')
            case 12:
                print("Dribbling the ball to the south")
                keyboard.press('s')
                time.sleep(dribble_duration)
                keyboard.release('s')
            case 13:
                print("Dribbling the ball to the southwest")
                keyboard.press('s')
                keyboard.press('a')
                time.sleep(dribble_duration)
                keyboard.release('a')
                keyboard.release('s')
            case 14:
                print("Dribbling the ball to the west")
                keyboard.press('a')
                time.sleep(dribble_duration)
                keyboard.release('a')
            case 15:
                print("Dribbling the ball to the northwest")
                keyboard.press('w')
                keyboard.press('a')
                time.sleep(dribble_duration)
                keyboard.release('a')
                keyboard.release('w')
            case 16:
                print("Shooting the ball")
                keyboard.press('p')
                time.sleep(shoot_duration)
                keyboard.release('p')
            case _:
                print("Unknown action:", action)
    def run(self):
        """
        Main execution loop for the controller. Captures screen frames,
        processes them with OCR and computer vision, and uses reinforcement
        learning to make gameplay decisions.
        """
        # Set up screen capture parameters for a wide screen monitor
        monitor = {"top": 0, "left": 0, "width": 2560, "height": 1600}
        frame_count = 0
        
        # Create directory for saving captured frames
        output_folder = "captured_frames"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with mss() as sct:
            print("Press 's' to start capturing, 'q' to quit")
            
            # Wait for start command
            while not keyboard.is_pressed('s'):
                time.sleep(0.1)
                if keyboard.is_pressed('q'):
                    print("Quitting...")
                    return
            
            # Start the game
            keyboard.press('enter')
            time.sleep(0.1)
            keyboard.release('enter')
            print("Starting capture cycle...")
            
            running = True
            self.temp_flag = False
            
            # Main capture and processing loop
            while running:
                # Capture current screen frame
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Skip frames if counter is active (e.g., after pausing)
                if self.skip_counter > 0:
                    self.skip_counter -= 1
                    frame_count += 1
                    continue

                # Handle temporary state flag for frame processing control
                if self.temp_flag == False:
                    # Save current frame and clear buffer
                    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
                    self.temp_flag = True
                    frame_count += 1
                    self.frames_to_send.clear()
                    
                    # Take action if one is queued
                    if self.action_taken is not None:
                        self.take_action(self.action_taken)
                        self.action_taken = None
                    continue
               
                # Process accumulated frames when buffer is full
                if len(self.frames_to_send) == 10:
                    # Pause game to process frames
                    self.pause_game()
                    
                    # Save frames for analysis
                    for i, frame in enumerate(self.frames_to_send):
                        cv2.imwrite(f"{output_folder}/frame_tbs_{frame_count+i:04d}.jpg", frame)
                    
                    # Get scene information from computer vision module
                    scene = self.computer_vision.run(self.frames_to_send, self.last_attempt)
                    print("scene:", scene)
                    self.frames_to_send.clear()
                    
                    # if start
                    # Reinforcement learning decision making
                    scene_toRL = scene.copy()
                    scene_toRL['Attemps'] = self.last_attempt
                    scene_toRL['Score'] = self.last_points
                    cur_state = get_state_from_positions(scene_toRL)
                    prev_state = self.rl_agent.prev_state 
                    action_taken = None
                    
                    # Handle first action or subsequent actions
                    if self.first_time:
                        self.first_time = False
                        action_taken = self.rl_agent.act(state=cur_state)
                        print("action_taken:", action_taken)
                        self.rl_agent.prev_state = cur_state
                    else:
                        action_taken = self.rl_agent.act(state=cur_state)
                        print("action_taken:", action_taken)
                        reward = calculate_reward(prev_state, cur_state, self.prev_action)
                        self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=False)
                    
                    # Periodically train the neural network
                    self.trainNN_counter += 1
                    if self.trainNN_counter == 64:
                        self.rl_agent.replay(32)  # Train on batch of 32 experiences
                        self.trainNN_counter = 0
                    
                    # Store actions for next iteration
                    self.action_taken = action_taken
                    self.prev_action = action_taken
                    # end if
                    
                    # Resume game and continue
                    self.resume_game()
                    continue
                
                # Add frame to buffer every 3rd frame to reduce processing load
                if frame_count % 3 == 0:
                    self.frames_to_send.append(frame)        
                    
                    # Extract game state with OCR
                    ocr_results = self.ocr_extractor.extract_points_and_attempts(frame)
                    
                    # Handle game restart condition
                    if self.last_attempt == '0' and ocr_results['attempts'] is None:
                        print("No attempts detected, skipping frame...", frame_count)
                        time.sleep(3)
                        keyboard.press('enter')
                        time.sleep(0.1)
                        keyboard.release('enter')
                        self.skip_counter = self.pause_frames_skip
                        
                        # Process end of game episode
                        scene_toRL = self.dummy_scene.copy()
                        scene_toRL['Attemps'] = self.last_attempt
                        scene_toRL['Score'] = self.last_points
                        cur_state = get_state_from_positions(scene_toRL)
                        prev_state = self.rl_agent.prev_state
                        self.last_attempt = None
                        self.temp_flag = False
                        frame_count += 1
                        
                        # Store terminal state transition for learning
                        reward = calculate_reward(prev_state, cur_state, self.prev_action)
                        self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                        self.first_time = True
                        continue

                    # Track attempts and handle attempt transitions
                    if self.last_attempt is None:
                        self.last_attempt = ocr_results['attempts']
                    else: 
                        if ocr_results['attempts'] != self.last_attempt:
                            # New attempt detected - process end of previous attempt
                            self.last_attempt = ocr_results['attempts']
                            scene_toRL = self.dummy_scene.copy()
                            scene_toRL['Attemps'] = self.last_attempt
                            scene_toRL['Score'] = self.last_points
                            
                            # Store transition with terminal state
                            cur_state = get_state_from_positions(scene_toRL)
                            prev_state = self.rl_agent.prev_state
                            reward = calculate_reward(prev_state, cur_state, self.prev_action)
                            self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                            self.first_time = True
                            
                            print("New attempt detected", frame_count)
                            self.skip_counter = self.between_attacks_skip
                            self.temp_flag = False
                            frame_count += 1
                            continue

                    # Print OCR results for debugging
                    print(f"\n----- OCR RESULTS FOR FRAME {frame_count+1} -----")
                    print(ocr_results)
                    print("----------------------------------------\n")
                    
                    # Check for quit command
                    if keyboard.is_pressed('q'):
                        print("Quit command detected, stopping...")
                        running = False
                        self.rl_agent.save("soccer_dqn_trained.pth")
                
                frame_count += 1
            
            # Print summary when finished
            print("FIFA Vision has stopped.")
            print("\n===== SUMMARY =====")
            print(f"Total frames processed: {frame_count}")
            print(f"Frames saved to: {output_folder}/")
            print("==================")

if __name__ == "__main__":
    # Create and run the controller
    controller = Controller()
    controller.run()