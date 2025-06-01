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
import vgamepad as vg
import threading
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

    def __init__(self, model_path=None, record_video=False):

        self.ocr_extractor = OCRExtractor()
        self.computer_vision = ML_ComputerVision()
        # self.computer_vision = ComputerVision()
        self.all_frames = []
        self.gamepad = vg.VDS4Gamepad()
        time.sleep(1) 
        
        self.pause_frames_skip = 5    
        self.between_attacks_skip = 5  
        self.first_start_skip = 120    
        self.skip_counter = 120         
        
        self.last_attempt = None       
        self.last_points = None        
        self.temp_flag = False          
        self.frames_to_send = []       
        self.first_time = True          
        self.is_paused = False         

        self.is_offside = False
        self.counter = 0            
        self.thread = None
        
        self.buttons = {
            'options': vg.DS4_BUTTONS.DS4_BUTTON_OPTIONS,
            'cross': vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE,
            'circle': vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE,
            'resume': vg.DS4_BUTTONS.DS4_BUTTON_CROSS,
            'north': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTH,
            'south': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTH,
            'east': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_EAST,
            'west': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_WEST,
            'nodir': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NONE,
            'northeast': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHEAST,
            'southeast': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHEAST,
            'southwest': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHWEST,
            'northwest': vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST,
        }
        

        self.to_be_sent = []
        self.rlstates=[]
        
        self.rl_agent = DQNAgent(load_path=model_path) if model_path else DQNAgent()
        self.trainNN_counter = 0        
        self.action_taken = None       
        self.prev_action = None 
        self.action_shoot=False

        self.hb_counter = 0
    
        self.dummy_scene = {
            "Attemps": None,
            "Score": None,
        }
        self.agent_y = None
        self.agent_x = None

        self.skip_2_frames = False

    def pause_game(self):
        self.gamepad.press_button(button=self.buttons['options'])
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(button=self.buttons['options'])
        self.gamepad.update()

    def resume_game(self,is_action=False):
        self.gamepad.press_button(button=self.buttons['resume'])
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(button=self.buttons['resume'])
        self.gamepad.update()
        if is_action:
            self.skip_counter = 0
            self.temp_flag = True
        else:
            self.skip_counter = self.pause_frames_skip
            self.temp_flag = False

    def movement_action(self, action):
        action = int(action)
        while self.stop_movement_flag == False:
            match action:
                case 8:
                    self.gamepad.directional_pad(self.buttons['north'])
                    self.gamepad.update()
                case 9:
                    self.gamepad.directional_pad(self.buttons['northeast'])
                    self.gamepad.update()
                case 10:
                    self.gamepad.directional_pad(self.buttons['east'])
                    self.gamepad.update()
                case 11:
                    self.gamepad.directional_pad(self.buttons['southeast'])
                    self.gamepad.update()
                case 12:
                    self.gamepad.directional_pad(self.buttons['south'])
                    self.gamepad.update()
                case 13:
                    self.gamepad.directional_pad(self.buttons['southwest'])
                    self.gamepad.update()
                case 14:
                    self.gamepad.directional_pad(self.buttons['west'])
                    self.gamepad.update()
                case 15:
                    self.gamepad.directional_pad(self.buttons['northwest'])
      
    def stop_current_thread(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_movement_flag = True
            self.thread.join(timeout=0.5) 
            if self.thread.is_alive():
                print("Thread lesa sha8al")
            self.thread = None
            
    def stop_any(self):
        self.gamepad.directional_pad(self.buttons['nodir'])
        self.gamepad.update()
        time.sleep(0.1)
    def make_action(self, action, pass_duration=0.25, dribble_duration=0.5, shoot_duration=0.18):
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
        
        match action:
            case 0:
                self.gamepad.directional_pad(self.buttons['north'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 1:
                self.gamepad.directional_pad(self.buttons['northeast'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 2:
                self.gamepad.directional_pad(self.buttons['east'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 3:
                self.gamepad.directional_pad(self.buttons['southeast'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 4:
                self.gamepad.directional_pad(self.buttons['south'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 5:
                self.gamepad.directional_pad(self.buttons['southwest'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 6:
                self.gamepad.directional_pad(self.buttons['west'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 7:
                self.gamepad.directional_pad(self.buttons['northwest'])
                self.gamepad.press_button(button=self.buttons['cross'])
                self.gamepad.update()
                time.sleep(pass_duration)
                self.gamepad.release_button(button=self.buttons['cross'])
                self.gamepad.update()
            case 8:
                self.gamepad.directional_pad(self.buttons['north'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 9:
                self.gamepad.directional_pad(self.buttons['northeast'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 10:
                self.gamepad.directional_pad(self.buttons['east'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 11:
                self.gamepad.directional_pad(self.buttons['southeast'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 12:
                self.gamepad.directional_pad(self.buttons['south'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 13:
                self.gamepad.directional_pad(self.buttons['southwest'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 14:
                self.gamepad.directional_pad(self.buttons['west'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 15:
                self.gamepad.directional_pad(self.buttons['northwest'])
                self.gamepad.update()
                time.sleep(dribble_duration)
            case 16:
                print("SHOOTING FROM THE NEAR POST")
                if np.random.rand() > 0.5:
                    print("SHOOTING DAY2A")
                    self.gamepad.directional_pad(self.buttons['east'])
                    self.gamepad.update()
                    time.sleep(0.2)
                    self.gamepad.press_button(button=self.buttons['circle'])
                    self.gamepad.update()
                    time.sleep(shoot_duration)
                    self.gamepad.release_button(button=self.buttons['circle'])
                    self.gamepad.update()
                    time.sleep(0.2)
                else:
                    print("SHOOTING B3EEDA")
                    self.gamepad.directional_pad(self.buttons['northeast'])
                    self.gamepad.update()
                    time.sleep(0.2)
                    self.gamepad.press_button(button=self.buttons['circle'])
                    self.gamepad.update()
                    time.sleep(shoot_duration)
                    self.gamepad.release_button(button=self.buttons['circle'])
                    self.gamepad.update()
                    time.sleep(0.2)
            case _:
                x = 1
                
    def run(self):       
        frame_count = 0

        gc.collect()
        

        with mss() as sct:
            monitor = sct.monitors[2]
            print("Press 'x' to start capturing, 'q' to quit")
            
            while not keyboard.is_pressed('x'):
                time.sleep(0.1)
                if keyboard.is_pressed('q'):
                    return
            
            self.gamepad.press_button(button=self.buttons['cross'])
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.release_button(button=self.buttons['cross'])
            self.gamepad.update()

            running = True
            self.temp_flag = False
            # self.start_time = time.time()
            
            # Main capture and processing loop
            while running:
                # current_time = time.time()
                # elapsed = current_time - self.start_time                
                # if self.is_paused:
                #         resume_process()
                #         self.is_paused = False
                try:
                    frame = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    print("FRAME ",self.counter)
                    self.counter += 1
                except MemoryError:
                    print("Memory Etmalet")
                    gc.collect()
                    exit(1)
                    continue
                self.all_frames.append(frame)
                
                ocr_results = self.ocr_extractor.extract_points_and_attempts(frame,self.counter - 1)
                print(ocr_results)

                if self.prev_action is not None:
                    if self.prev_action in range(8,16):
                        self.stop_movement_flag = False
                        self.thread = threading.Thread(target=self.movement_action, args=(self.prev_action,))
                        self.thread.start()
                    else: 
                        self.stop_movement_flag = False
                        self.thread = threading.Thread(target=self.movement_action, args=(10,))
                        self.thread.start()                

                if ocr_results['points'] is not None and self.last_attempt is not None and self.last_attempt != '0':
                    if self.last_points is not None and (ocr_results['points'] >= self.last_points):
                        self.last_points = ocr_results['points']
                # if ocr_results['points'] is not None and self.last_attempt is not 0:
                #     self.last_points = ocr_results['points']
                if ocr_results['points'] is not None and self.last_points is not None and (ocr_results['points'] > self.last_points) and self.last_attempt == '0':
                    self.last_points = ocr_results['points']
                if True:
                    if self.skip_counter > 0:
                        self.skip_counter -= 1
                        print("Skipping Frame: ",frame_count)
                        frame_count += 1
                        continue

                if True:                   
                    if self.temp_flag == False:
                        print("TEMP FLAG:",self.temp_flag)              
                        self.temp_flag = True
                        frame_count += 1
                        self.frames_to_send.clear()
                        continue
               
                print("BUFFER SIZE:", len(self.frames_to_send))
                if len(self.frames_to_send) == 3:
                    self.to_be_sent.append(self.frames_to_send.copy())
                    print("PAUSE")
                    self.pause_game()
                    # pause_process() De el original bs b3d ma fifa bazeet 8irna l pause game
                    
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
                            # self.stop_movement_action(self.prev_action)
                            self.stop_current_thread()
                            
                            self.stop_movement_flag = False
                            self.thread = threading.Thread(target=self.movement_action, args=(10,))
                            self.thread.start()
                            # self.movement_action(10)
                    else:
                        self.hb_counter = 0

                    if scene is None and has_ball is None:
                        
                        print("OFFSIDE DETECTED ")
                        self.frames_to_send.clear()
                        gc.collect()
                        self.is_paused = False
                        self.is_offside = True
                        # self.rl_agent.memory.pop()
                        # resume_process()    
                        self.resume_game()
                        
                        print("RESUMING")
                        continue

                    print("\nHAS BALL:", has_ball,"\n")


                    tmp_store = self.frames_to_send[-1].copy()

                    self.frames_to_send.clear()
                    gc.collect()
                    
                    if has_ball and not self.action_shoot:
                        # if self.prev_action is not None:
                        #     self.stop_movement_action(self.prev_action)
                        # else:
                        #     self.stop_movement_action(10)    
                        self.stop_current_thread()
                        scene_toRL = scene.copy()
                        scene_toRL['Attemps'] = self.last_attempt
                        scene_toRL['Score'] = self.last_points
                        self.rlstates.append(scene_toRL)
                        cur_state = get_state_from_positions(scene_toRL)

                        # if self.prev_action is not None and self.prev_action in range(0,8) and self.skip_2_frames:

                        #     self.frames_to_send.append(tmp_store)
                        #     self.is_paused = False
                        #     self.skip_2_frames = False
                        #     # resume_process()
                        #     self.resume_game()
                        #     print("RESUMING")
                        #     continue
                        # self.agent_y = cur_state[-1] * 90
                        # self.agent_x = cur_state[-2] * 60
                        # print("\nAGENT Y:", self.agent_y,"\n")

                        prev_state = self.rl_agent.prev_state 
                        action_taken = None
                        
                        if self.first_time:
                            self.first_time = False
                            action_taken = self.rl_agent.act(state=cur_state)
                            self.rl_agent.prev_state = cur_state
                        else:
                            action_taken = self.rl_agent.act(state=cur_state)
                            # compute=compute_reward(prev_state, cur_state, self.prev_action)
                            reward = calculate_reward(prev_state, cur_state, self.prev_action)
                            # action mapped using action_descriptions
                            action_description = action_descriptions.get(self.prev_action, f"Unknown action: {self.prev_action}")
                            print("\nREWARD NEW ACT:", reward," action: ",action_description,"\n")

                            if self.is_offside:
                                self.is_offside = False
                            else:
                                if self.prev_action is not None:
                                    self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=False)
                                # self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=False) OUDSAA
                            self.rl_agent.prev_state = cur_state #OUDA

                        # if len(self.rl_agent.memory) % 64 == 0 or (len(self.rl_agent.memory) > 1000 and len(self.rl_agent.memory) % 16 == 0):
                        #     self.rl_agent.replay(64) 
                        print("PRESS L TO TAKE ACTION")
                        keyboard.wait('l')
                        
                        print("\nACTION TAKEN:", action_taken,"\n")
                        self.action_taken = action_taken
                        if action_taken in range(0,8):
                            self.skip_2_frames = True
                        if action_taken == 16:
                            self.action_shoot = True
                        self.prev_action = action_taken
                    else:
                        time.sleep(0.4)
                    

                    print("RESSSSUMING")
                    if self.action_taken is not None:
                        self.is_paused = False
                        # resume_process()
                        self.resume_game()
                        # self.make_action(10)
                        time.sleep(0.2)
                        self.make_action(self.action_taken)
                        self.action_taken = None
                    else:
                        self.is_paused = False
                        # resume_process()
                        self.resume_game()
                    continue
                

                # if frame_count % 2 == 0:
                #     frame_count += 1
                #     continue
                
                self.frames_to_send.append(frame)
                                
                # Handle game restart condition
                if self.last_attempt == '0' and ocr_results['attempts'] is None:

                    self.stop_current_thread()
                    time.sleep(3)
                    self.gamepad.press_button(button=self.buttons['cross'])
                    self.gamepad.update()
                    time.sleep(0.1)
                    self.gamepad.release_button(button=self.buttons['cross'])
                    self.gamepad.update()
                    self.skip_counter = self.pause_frames_skip
                    
                    scene_toRL = self.dummy_scene.copy()
                    scene_toRL['Attemps'] = self.last_attempt
                    scene_toRL['Score'] = self.last_points
                    cur_state = get_state_from_positions(scene_toRL)
                    prev_state = self.rl_agent.prev_state
                    self.last_attempt = None
                    self.last_points = None
                    self.temp_flag = False
                    frame_count += 1
                    reward = calculate_reward(prev_state, cur_state, self.prev_action)
                    action_description = action_descriptions.get(self.prev_action, f"Unknown action: {self.prev_action}")
                    print("REWARD NEW GAME:", reward," ",action_description,"\n")
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
                    self.hb_counter = 0
                    continue

                if self.last_attempt is None:
                    self.last_attempt = ocr_results['attempts']
                    self.last_points = ocr_results['points']
                else: 
                    if ocr_results['attempts'] != self.last_attempt and ocr_results['attempts'] is not None:
                        # self.stop_movement_action(self.prev_action)
                        self.stop_current_thread()
                        self.computer_vision.tracker.reset_tracker()
                        # attempt should be -1 only form last attempt
                        if ocr_results['attempts'] == str(int(self.last_attempt) - 1):
                        # if ocr_results['attempts'] <= self.last_attempt:
                            self.last_attempt = ocr_results['attempts']
                        self.last_points = ocr_results['points']
                        scene_toRL = self.dummy_scene.copy()
                        scene_toRL['Attemps'] = self.last_attempt
                        scene_toRL['Score'] = self.last_points
                        
                        cur_state = get_state_from_positions(scene_toRL)
                        prev_state = self.rl_agent.prev_state
                        reward = calculate_reward(prev_state, cur_state, self.prev_action)
                        action_description = action_descriptions.get(self.prev_action, f"Unknown action: {self.prev_action}")
                        print("\nREWARD NEW ATTEM:", reward," ",action_description,"\n")
                        if self.is_offside:
                            self.is_offside = False
                        else:
                            if self.prev_action is not None:
                                self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                            # self.rl_agent.remember(prev_state, self.prev_action, reward, cur_state, done=True)
                        self.first_time = True                        
                        self.skip_counter = self.between_attacks_skip
                        self.temp_flag = False
                        frame_count += 1
                        self.action_shoot= False
                        self.prev_action = None
                        self.hb_counter = 0
                        continue


                
                if keyboard.is_pressed('q'):
                    running = False
                    model_save_path = file_name
                    self.rl_agent.save(model_save_path)
                    self.stop_current_thread()
                    
                    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frames_dir = os.path.join(os.getcwd(), f"frames_{run_timestamp}")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    for i, frame in enumerate(self.all_frames):
                        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                        cv2.imwrite(frame_path, frame)
                        
                    print(f"Saved {len(self.all_frames)} frames to {frames_dir}")
                    
                    
                                   
                frame_count += 1
            
          

if __name__ == "__main__":
    controller = Controller(
        # file_name
        )
    rl_model = controller.rl_agent
    try:
        controller.run()
    except Exception as e:
        model_save_path = file_name
        rl_model.save(model_save_path)
        print("Saved model to", model_save_path)

        print(f"An error occurred: {e}")
        print("\033[91m")
        traceback.print_exc()
        print("\033[0m")


