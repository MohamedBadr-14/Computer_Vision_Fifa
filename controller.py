import time
import numpy as np
import cv2
import keyboard
import os
import logging
from datetime import datetime
from mss import mss
from ocr_extractor import OCRExtractor
# from computer_vision import ComputerVision
from rl import DQNAgent, get_state_from_positions, calculate_reward,compute_reward
from pr import pause_process, resume_process
import gc
import traceback
from computer_vision import ML_ComputerVision




file_name="MODEL_21_5.pth"


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
            16: "Shooting the ball"
        }
class Controller:
    """
    Main controller class for an AI-powered soccer game automation system.
    This class integrates computer vision, OCR, and reinforcement learning to
    autonomously play a soccer game.
    """
    def __init__(self, model_path=None, record_video=False):
        """
        Initialize the controller with various components and parameters.
        
        Args:
            model_path (str, optional): Path to a pre-trained DQN model. If None, a new model is created.
            record_video (bool, optional): Whether to record gameplay with logs as a video.
        """
 
        
        # Initialize OCR and computer vision modules
        self.ocr_extractor = OCRExtractor()
        self.computer_vision = ML_ComputerVision()
        # self.logger.info("OCR and Computer Vision modules initialized")
        self.all_frames = []
        
        # Frame skipping parameters for game control
        self.pause_frames_skip = 10    # Number of frames to skip after pausing 10
        self.between_attacks_skip = 5  # Frames to skip between game attacks 15
        self.first_start_skip = 110    # Frames to skip at initial game start
        self.skip_counter = 110         # Counter for skipping frames
        
        # Game state tracking
        self.last_attempt = None        # Track the last attempt number
        self.last_points = None         # Track the last score points
        self.temp_flag = False          # Flag for temporary state control
        self.frames_to_send = []        # Buffer for frames to process
        self.first_time = True          # Flag for first action
        self.is_paused = False          # Flag for game pause state

        self.is_offside = False        # Flag for offside detection
        

        self.to_be_sent = []
        self.rlstates=[]
        
        # Reinforcement learning setup
        # self.logger.info(f"Initializing DQN Agent with model path: {model_path if model_path else 'None - creating new model'}")
        self.rl_agent = DQNAgent(load_path=model_path) if model_path else DQNAgent()
        self.trainNN_counter = 0        # Counter for neural network training
        self.action_taken = None        # Current action
        self.prev_action = None         # Previous action
        self.action_shoot=False

        self.hb_counter = 0
    
        # Default scene state when data is unavailable
        self.dummy_scene = {
            "Attemps": None,
            "Score": None,
        }
        self.agent_y = None
        self.agent_x = None

        self.skip_2_frames = False

    def pause_game(self):
        """
        Pause the game by simulating ESC key press.
        """
        # self.logger.info("Pausing game")
        keyboard.press('esc')
        time.sleep(0.1)
        keyboard.release('esc')

    def resume_game(self,is_action=False):
        """
        Resume the game by simulating ENTER key press and reset relevant counters.
        """
        # self.logger.info("Resuming game")
        keyboard.press('enter')
        time.sleep(0.1)
        keyboard.release('enter')
        if is_action:
            self.skip_counter = 0
            self.temp_flag = True
        else:
            self.skip_counter = self.pause_frames_skip
            self.temp_flag = False
        # self.skip_counter = self.pause_frames_skip
        # self.temp_flag = False
        # self.logger.debug(f"Reset skip_counter to {self.pause_frames_skip}")
    def movement_action(self, action):
        match action:
            case 8:
                keyboard.press('w')
                keyboard.press('d')
            case 9:
                keyboard.press('w')
                keyboard.press('d')
            case 10:
                keyboard.press('d')
            case 11:
                keyboard.press('s')
                keyboard.press('d')
            case 12:
                keyboard.press('s')

            case 13:
                keyboard.press('s')
                keyboard.press('a')

            case 14:
                keyboard.press('a')

            case 15:
                keyboard.press('w')
                keyboard.press('a')


    def stop_movement_action(self, action):
        match action:
            case 0:
                # release all keys
                keyboard.release('w')
                keyboard.release('d')
                keyboard.release('s')
                keyboard.release('a')
            case 8:
                keyboard.release('w')
                keyboard.release('d')
            case 9:
                keyboard.release('w')
                keyboard.release('d')
            case 10:
                keyboard.release('d')
            case 11:
                keyboard.release('s')
                keyboard.release('d')
            case 12:
                keyboard.release('s')

            case 13:
                keyboard.release('s')
                keyboard.release('a')

            case 14:
                keyboard.release('a')

            case 15:
                keyboard.release('w')
                keyboard.release('a')
            case _:
                x = 1
                # self.logger.warning(f"Unknown action: {action}")           

    def make_action(self, action, pass_duration=0.25, dribble_duration=0.5, shoot_duration=0.225):
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
            16: "Shooting the ball"
        }
        
        # action_description = action_descriptions.get(action, f"Unknown action: {action}")
        # self.logger.info(f"Making action [{action}]: {action_description}")
        self.stop_movement_action(0)
        match action:
            case 0:
                keyboard.press('w')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('w')
            case 1:
                keyboard.press('w')
                keyboard.press('d')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('d')
                keyboard.release('w')
            case 2:
                keyboard.press('d')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('d')
            case 3:
                keyboard.press('s')
                keyboard.press('d')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('d')
                keyboard.release('s')
            case 4:
                keyboard.press('s')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('s')
            case 5:
                keyboard.press('s')
                keyboard.press('a')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('a')
                keyboard.release('s')
            case 6:
                keyboard.press('a')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('a')
            case 7:
                keyboard.press('w')
                keyboard.press('a')
                keyboard.press('p')
                time.sleep(pass_duration)
                keyboard.release('p')
                keyboard.release('a')
                keyboard.release('w')
            case 8:
                keyboard.press('w')
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
                keyboard.release('w')
            case 9:
                keyboard.press('w')
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
                keyboard.release('w')
            case 10:
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
            case 11:
                keyboard.press('s')
                keyboard.press('d')
                time.sleep(dribble_duration)
                keyboard.release('d')
                keyboard.release('s')
            case 12:
                keyboard.press('s')
                time.sleep(dribble_duration)
                keyboard.release('s')
            case 13:
                keyboard.press('s')
                keyboard.press('a')
                time.sleep(dribble_duration)
                keyboard.release('a')
                keyboard.release('s')
            case 14:
                keyboard.press('a')
                time.sleep(dribble_duration)
                keyboard.release('a')
            case 15:
                keyboard.press('w')
                keyboard.press('a')
                time.sleep(dribble_duration)
                keyboard.release('a')
                keyboard.release('w')
            case 16:
                # we have agent_y from 0 to 90 
                # we have the far post @ 49 and the near post @ 41
                # self.stop_movement_action(0)
                if self.agent_y > 49:
                    print("SHOOTING FROM THE FAR POST")
                    # shoot from above the far post
                    # random between 2 directions
                    if np.random.rand() > 0.5:
                        print("SHOOTING DAY2A")
                        keyboard.press('d')
                        time.sleep(0.2)
                        keyboard.press('k')
                        time.sleep(shoot_duration)
                        keyboard.release('k')
                        time.sleep(0.2)
                        keyboard.release('d')
                    else:
                        print("SHOOTING B3EEDA")
                        keyboard.press('d')
                        keyboard.press('s')
                        time.sleep(0.2)
                        keyboard.press('k')
                        time.sleep(shoot_duration)
                        keyboard.release('k')
                        time.sleep(0.2)
                        keyboard.release('s')
                        keyboard.release('d')
                elif self.agent_y < 41:
                    print("SHOOTING FROM THE NEAR POST")
                    if np.random.rand() > 0.5:
                        print("SHOOTING DAY2A")
                        keyboard.press('d')
                        time.sleep(0.2)
                        keyboard.press('k')
                        time.sleep(shoot_duration)
                        keyboard.release('k')
                        time.sleep(0.2)
                        keyboard.release('d')
                    else:
                        print("SHOOTING B3EEDA")
                        keyboard.press('d')
                        keyboard.press('w')
                        time.sleep(0.2)
                        keyboard.press('k')
                        time.sleep(shoot_duration)
                        keyboard.release('k')
                        time.sleep(0.2)
                        keyboard.release('w')
                        keyboard.release('d')
                    # shoot from below the near post
                else:
                    print("SHOOTING FROM THE MIDDLE")
                    if np.random.rand() > 0.5:
                        print("SHOOTING B3EEDA")
                        keyboard.press('d')
                        keyboard.press('w')
                        time.sleep(0.2)
                        keyboard.press('k')
                        time.sleep(shoot_duration)
                        keyboard.release('k')
                        time.sleep(0.2)
                        keyboard.release('w')
                        keyboard.release('d')


                    else:
                        print("SHOOTING DAY2A")
                        keyboard.press('d')
                        keyboard.press('s')
                        time.sleep(0.2)
                        keyboard.press('k')
                        time.sleep(shoot_duration)
                        keyboard.release('k')
                        time.sleep(0.2)
                        keyboard.release('s')
                        keyboard.release('d')
                    # shoot from the middle
                # keyboard.press('d')
                # keyboard.press('k')
                # time.sleep(shoot_duration)
                # keyboard.release('k')
                # keyboard.release('d')
                # keyboard.wait('l')
            case _:
                x = 1
                # self.logger.warning(f"Unknown action: {action}")
                
        # self.logger.debug(f"Action {action} completed")

    def run(self):
        """
        Main execution loop for the controller. Captures screen frames,
        processes them with OCR and computer vision, and uses reinforcement
        learning to make gameplay decisions.
        """
        # Set up screen capture parameters for a wide screen monitor
        monitor = {"top": 0, "left": 0, "width": 2560, "height": 1600}
        frame_count = 0

        gc.collect()  # Force garbage collection to free up memory
        

        with mss() as sct:
            print("Press 's' to start capturing, 'q' to quit")
            
            # Wait for start command
            while not keyboard.is_pressed('s'):
                time.sleep(0.1)
                if keyboard.is_pressed('q'):
                
                    # self.logger.info("Quit command detected before starting")
                    return
            
            # Start the game
            keyboard.press('enter')
            time.sleep(0.1)
            keyboard.release('enter')
            # self.logger.info("Starting capture cycle...")
            
            running = True
            self.temp_flag = False
            # self.start_time = time.time()
            
            # Main capture and processing loop
            while running:
                # current_time = time.time()
                # elapsed = current_time - self.start_time                
                # Capture current screen frame
                # if self.is_paused:
                #         resume_process()
                #         self.is_paused = False
                try:
                    frame = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                except MemoryError:
                    print("MemoryError: Frame capture failed. Attempting to free up memory.")
                    gc.collect()  # Force garbage collection to free up memory
                    exit(1)
                    continue
                # self.all_frames.append(frame)
                
                # Extract OCR results for logging
                ocr_results = self.ocr_extractor.extract_points_and_attempts(frame)
                print(ocr_results)

                if self.prev_action is not None:
                    if self.prev_action in range(8,16):
                        self.movement_action(self.prev_action)
                    else: 
                        self.movement_action(10)
                

                if ocr_results['points'] is not None and self.last_attempt is not None and self.last_attempt != '0':
                    self.last_points = ocr_results['points']
                # if ocr_results['points'] is not None and self.last_attempt is not 0:
                #     self.last_points = ocr_results['points']
                if ocr_results['points'] is not None and self.last_points is not None and (ocr_results['points'] > self.last_points) and self.last_attempt == '0':
                    self.last_points = ocr_results['points']
                # Skip frames if counter is active (e.g., after pausing)

                if True:
                    
                    if self.skip_counter > 0:
                        self.skip_counter -= 1
                        frame_count += 1
                        continue

                # Handle temporary state flag for frame processing control
                if True:
                   
                    if self.temp_flag == False:
                        print("TEMP FLAG:",self.temp_flag)              
                        self.temp_flag = True
                        frame_count += 1
                        self.frames_to_send.clear()
                        continue
               
                # Process accumulated frames when buffer is full
                print("BUFFER SIZE:", len(self.frames_to_send))
                if len(self.frames_to_send) == 3:
                    self.to_be_sent.append(self.frames_to_send.copy())
                    print("PAUSE")
                    pause_process()
                    print("PAUSEDDDD")
                    
                    time_before = time.time()
                    time.sleep(0.1)
                    self.is_paused = True

                    scene, has_ball= self.computer_vision.run(self.frames_to_send, self.last_attempt)
                    time_after = time.time()
                    print("Processing time: ", time_after - time_before)
                    if not has_ball:
                        if self.prev_action is not None and self.prev_action in range(0,8):
                            self.skip_2_frames = False
                        self.hb_counter += 1
                        if self.hb_counter == 2:
                            self.hb_counter = 0
                            self.stop_movement_action(self.prev_action)
                            self.movement_action(10)
                    else:
                        self.hb_counter = 0

                    if scene is None and has_ball is None:
                        
                        print("\033[1;31;7m" + 
                                "################################\n" +
                                "#                              #\n" +
                                "#      OFFSIDE DETECTED        #\n" +
                                "#                              #\n" +
                                "################################" +
                                "\033[0m")
                        self.frames_to_send.clear()
                        gc.collect()
                        self.is_paused = False
                        self.is_offside = True
                        # self.rl_agent.memory.pop()
                        resume_process()    
                        print("RESUMING")
                        continue

                    print("\nHAS BALL:", has_ball,"\n")


                    tmp_store = self.frames_to_send[-1].copy()

                    self.frames_to_send.clear()
                    gc.collect()
                    
                    if has_ball and not self.action_shoot:
                        if self.prev_action is not None:
                            self.stop_movement_action(self.prev_action)
                        else:
                            self.stop_movement_action(10)    
                        # Reinforcement learning decision making
                        scene_toRL = scene.copy()
                        scene_toRL['Attemps'] = self.last_attempt
                        scene_toRL['Score'] = self.last_points
                        self.rlstates.append(scene_toRL)
                        cur_state = get_state_from_positions(scene_toRL)
                        # print("\n Prev x:", self.agent_x,"\n")
                        # print("\n Prev y:", self.agent_y,"\n")
                        # print("\n Cur x:", cur_state[-2]*60,"\n")
                        # print("\n Cur y:", cur_state[-1]*90,"\n")
                        if self.prev_action is not None and self.prev_action in range(0,8) and self.skip_2_frames:
                            # keyboard.wait('l')
                            # prev_x = self.agent_x
                            # prev_y = self.agent_y
                            # curr_x = cur_state[-2] * 60
                            # curr_y = cur_state[-1] * 90
                            # # if the distance is less than 10, we are in the same position
                            # if abs(prev_x - curr_x) < 5 and abs(prev_y - curr_y) < 2.5:
                            #     print("\033[1;33;40m" +  # Bold yellow text on black background
                            #         "╔═════════════════════════════════╗\n" +
                            #         "║   WE ARE IN THE SAME POSITION   ║\n" +
                            #         "╚═════════════════════════════════╝" +
                            #         "\033[0m") 
                            self.frames_to_send.append(tmp_store)
                            self.is_paused = False
                            self.skip_2_frames = False
                            resume_process()
                            print("RESUMING")
                            continue
                        self.agent_y = cur_state[-1] * 90
                        self.agent_x = cur_state[-2] * 60
                        print("\nAGENT Y:", self.agent_y,"\n")

                        prev_state = self.rl_agent.prev_state 
                        action_taken = None
                        
                        # Handle first action or subsequent actions
                        if self.first_time:
                            self.first_time = False
                            action_taken = self.rl_agent.act(state=cur_state)
                            # self.logger.info(f"First action taken: {action_taken}")
                            self.rl_agent.prev_state = cur_state
                        else:
                            action_taken = self.rl_agent.act(state=cur_state)
                            # self.logger.info(f"Action taken: {action_taken}")
                            # compute=compute_reward(prev_state, cur_state, self.prev_action)
                            reward = calculate_reward(prev_state, cur_state, self.prev_action)
                            # action mapped using action_descriptions
                            action_description = action_descriptions.get(self.prev_action, f"Unknown action: {self.prev_action}")
                            print("\nREWARD NEW ACT:", reward," action: ",action_description,"\n")
                            # print("\033[1;31;7m" + 
                            #     "################################\n" +
                            #     "#                              #\n" +
                            #     f'#      {compute}        #\n' +
                            #     "#                              #\n" +
                            #     "################################" +
                            #     "\033[0m")
                            # self.logger.info(f"Calculated reward: {reward}")
                            if self.is_offside:
                                self.is_offside = False
                            else:
                                if self.prev_action is not None:
                                    self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=False)
                                # self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=False) OUDSAA
                            self.rl_agent.prev_state = cur_state #OUDA

                        # if len(self.rl_agent.memory) % 64 == 0 or (len(self.rl_agent.memory) > 1000 and len(self.rl_agent.memory) % 16 == 0):
                        #     self.rl_agent.replay(64)  # Train on batch of 32 experiences
                   
                        print("\nACTION TAKEN:", action_taken,"\n")
                        self.action_taken = action_taken
                        if action_taken in range(0,8):
                            self.skip_2_frames = True
                        if action_taken == 16:
                            self.action_shoot = True
                        self.prev_action = action_taken
                    else:
                        time.sleep(0.4)
                    
                    # Resume game and continue
                    # is_action = True if (self.action_taken is not None and has_ball) else False
                    # wait for key d pressed
                    # self.logger.info("Waiting for 'd' key press to resume game...")
                    # keyboard.wait('l')
                    print("RESSSSUMING")
                    if self.action_taken is not None:
                        self.is_paused = False
                        resume_process()
                        # self.make_action(10)
                        self.make_action(self.action_taken)
                        self.action_taken = None
                    else:
                        self.is_paused = False
                        resume_process()
                    continue
                

                # if frame_count % 2 == 0:
                #     # Skip every other frame for processing
                #     frame_count += 1
                #     continue
                # Add frame to buffer
                self.frames_to_send.append(frame)
                # self.logger.debug(f"Added frame to buffer, current size: {len(self.frames_to_send)}/5, frame count: {frame_count}")
                
                # Record frame in video
                # self.write_frame_to_video(frame, frame_count, ocr_results)
                
                # Handle game restart condition
                if self.last_attempt == '0' and ocr_results['attempts'] is None:
                    # self.logger.info(f"No attempts detected at frame {frame_count}, attempting restart")
                    self.stop_movement_action(self.prev_action)
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
                    self.last_points = None
                    self.temp_flag = False
                    frame_count += 1
                    
                    # Store terminal state transition for learning
                    # compute=compute_reward(prev_state, cur_state, self.prev_action)
                    # print("\033[1;31;7m" + 
                    #             "################################\n" +
                    #             "#                              #\n" +
                    #             f'#      {compute}        #\n' +
                    #             "#                              #\n" +
                    #             "################################" +
                    #             "\033[0m")

                    reward = calculate_reward(prev_state, cur_state, self.prev_action)
                    action_description = action_descriptions.get(self.prev_action, f"Unknown action: {self.prev_action}")
                    print("\nREWARD NEW GAME:", reward," ",action_description,"\n")
                    # self.logger.info(f"End of episode reward: {reward}")
                    if self.is_offside:
                        self.is_offside = False
                    else:
                        if self.prev_action is not None:
                            self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                        # self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                    self.first_time = True
                    self.action_shoot= False
                    self.computer_vision.tracker.reset_tracker()
                    self.prev_action = None
                    self.prev_farthest_team_1 = None
                    self.prev_farthest_team_2 = None
                    self.prev_ball_pos = None
                    self.hb_counter = 0
                    continue

                # Track attempts and handle attempt transitions
                if self.last_attempt is None:
                    self.last_attempt = ocr_results['attempts']
                    self.last_points = ocr_results['points']
                    # self.logger.info(f"Initial attempt detected: {self.last_attempt}")
                else: 
                    if ocr_results['attempts'] != self.last_attempt and ocr_results['attempts'] is not None:
                        # New attempt detected - process end of previous attempt
                        self.stop_movement_action(self.prev_action)
                        self.computer_vision.tracker.reset_tracker()
                        self.last_attempt = ocr_results['attempts']
                        self.last_points = ocr_results['points']
                        scene_toRL = self.dummy_scene.copy()
                        scene_toRL['Attemps'] = self.last_attempt
                        scene_toRL['Score'] = self.last_points
                        
                        # Store transition with terminal state
                        cur_state = get_state_from_positions(scene_toRL)
                        prev_state = self.rl_agent.prev_state
                        # compute=compute_reward(prev_state, cur_state, self.prev_action)
                        # print("\033[1;31;7m" + 
                        #         "################################\n" +
                        #         "#                              #\n" +
                        #         f'#      {compute}        #\n' +
                        #         "#                              #\n" +
                        #         "################################" +
                        #         "\033[0m")

                        reward = calculate_reward(prev_state, cur_state, self.prev_action)
                        action_description = action_descriptions.get(self.prev_action, f"Unknown action: {self.prev_action}")
                        print("\nREWARD NEW ATTEM:", reward," ",action_description,"\n")
                        # self.logger.info(f"End of attempt reward: {reward}")
                        if self.is_offside:
                            self.is_offside = False
                        else:
                            if self.prev_action is not None:
                                self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                            # self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                        self.first_time = True
                        
                        # self.logger.info(f"Skipping {self.between_attacks_skip} frames between attempts")
                        self.skip_counter = self.between_attacks_skip
                        self.temp_flag = False
                        frame_count += 1
                        self.action_shoot= False
                        self.prev_action = None
                        self.prev_farthest_team_1 = None
                        self.prev_farthest_team_2 = None
                        self.prev_ball_pos = None
                        self.hb_counter = 0
                        continue


                
                # Check for quit command
                if keyboard.is_pressed('q'):
                    # self.logger.info("Quit command detected at elapsed time: {:.2f}s".format(elapsed))
                    running = False
                    model_save_path = file_name
                    # self.logger.info(f"Saving DQN model to {model_save_path}")
                    self.rl_agent.save(model_save_path)
                    
                    for i in range(8,16):
                        self.stop_movement_action(i) 
                    # save self.to_be_sent to a file

                    if self.is_paused:
                        resume_process()
                        self.is_paused = False
                    
                    
                    # to_be_sent_array = np.array(self.to_be_sent)
                    # to_be_sent_path = "to_be_sent.npy"
                    # np.save(to_be_sent_path, to_be_sent_array)

                    
                    
                    #  loop over self.all_frames and save them to a folder
                    # all_frames_path = "all_frames"
                    # os.makedirs(all_frames_path, exist_ok=True)
                    # for i, frame in enumerate(self.all_frames):
                    #     frame_filename = os.path.join(all_frames_path, f'frame_{i:04d}.png')
                    #     cv2.imwrite(frame_filename, frame)
                
                frame_count += 1
            
          

if __name__ == "__main__":
    # Create and run the controller with optional video recording
    controller = Controller(
        # file_name
        )
    rl_model = controller.rl_agent
    # controller.run()
    try:
        controller.run()
    except Exception as e:
        model_save_path = file_name
        rl_model.save(model_save_path)
        print("Saved model to", model_save_path)

        print(f"An error occurred: {e}")
        print("\033[91m")  # Red text for error
        traceback.print_exc()
        print("\033[0m")
        try:
            resume_process()
        except Exception as e:
            print(f"An error occurred while resuming the process: {e}")
            print("\033[91m")  # Red text for error
            traceback.print_exc()
            print("\033[0m")


