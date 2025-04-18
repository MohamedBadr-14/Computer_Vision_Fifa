import time
import numpy as np
from mss import mss
import cv2
import keyboard 
import os
from ocr_extractor import OCRExtractor
from computer_vision import ComputerVision

class Controller:
    def __init__(self):
        self.ocr_extractor = OCRExtractor()
        self.pause_frames_skip = 20 #clear number of frames
        self.between_attacks_skip = 10
        self.first_start_skip = 110
        self.skip_counter = 110
        self.last_attempt = None
        self.last_points = None
        self.temp_flag = False
        self.frames_to_send = []
        self.computer_vision = ComputerVision()



    def pause_game(self):
        keyboard.press('esc')
        time.sleep(0.1)
        keyboard.release('esc')


    def resume_game(self):
        keyboard.press('enter')
        time.sleep(0.1)
        keyboard.release('enter')
        self.skip_counter = self.pause_frames_skip
        self.temp_flag = False
        

        
    def run(self):
        monitor = {"top": 0, "left": 0, "width": 2560, "height": 1600}  # Adjust for Legion 5 Pro
        frame_count = 0
        
        # Create output folder for saving frames if needed
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
            
            keyboard.press('enter')
            time.sleep(0.1)
            keyboard.release('enter')
            print("Starting capture cycle...")
            running = True
            self.temp_flag = False
            while running:
             
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                

                if self.skip_counter > 0:
                    self.skip_counter -= 1
                    frame_count += 1
                    continue

                if self.temp_flag == False:
                    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
                    self.temp_flag = True
                    frame_count += 1
                    self.frames_to_send.clear()
                    continue
               
                if len(self.frames_to_send) == 10:
                    # Send frames to the server
                    # send_frames_to_server(self.frames_to_send)
                    # Clear the list after sending
                    self.pause_game()
                    self.computer_vision.run(self.frames_to_send,self.last_attempt)
                    self.frames_to_send.clear()
                    self.resume_game()



                if frame_count % 3 == 0:
                    self.frames_to_send.append(frame)        
                    # 2. Run OCR extraction
                    # print(f"Running OCR on frame {frame_count+1}...")
                    ocr_results = self.ocr_extractor.extract_points_and_attempts(frame)
                    # print("last_attempt:", self.last_attempt)
                    # print("ocr_results['attempts']:", ocr_results['attempts'])
                    if self.last_attempt == '0' and ocr_results['attempts'] is None:
                        print("No attempts detected, skipping frame...",frame_count)
                        time.sleep(3)
                        keyboard.press('enter')
                        time.sleep(0.1)
                        keyboard.release('enter')
                        self.skip_counter = self.pause_frames_skip
                        self.last_attempt = None
                        self.temp_flag = False
                        frame_count += 1
                        continue

                    if self.last_attempt == None:
                        self.last_attempt = ocr_results['attempts']
                    else: 
                        if ocr_results['attempts'] != self.last_attempt:
                            self.last_attempt = ocr_results['attempts']
                            print("New attempt detected",frame_count)
                            # 4. Save the output image with OCR results
                            self.skip_counter = self.between_attacks_skip
                            self.temp_flag = False
                            frame_count += 1
                            continue

                    
                    # 3. Print OCR results
                    print(f"\n----- OCR RESULTS FOR FRAME {frame_count+1} -----")
                    print(ocr_results)
                    print("----------------------------------------\n")
                    
                    # 6. Wait before next capture
                    # time.sleep(1)  # Wait 1 second before next capture
                    
                    # 7. Increment frame counter
                    
                    
                    # Check for quit command
                    if keyboard.is_pressed('q'):
                        print("Quit command detected, stopping...")
                        running = False
                
                

                frame_count += 1
            
            print("FIFA Vision has stopped.")
            
            # After all frames are processed, print summary
            print("\n===== SUMMARY =====")
            print(f"Total frames processed: {frame_count}")
            print(f"Frames saved to: {output_folder}/")
            print("==================")

if __name__ == "__main__":


    controller = Controller()
    controller.run()