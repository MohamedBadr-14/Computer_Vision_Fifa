import time
import mss
from PIL import Image
import keyboard
from computer_vision import ML_ComputerVision 
import time
from rm_mem import *
import cv2
from controller_rl import Controller

def capture_screen() -> Image.Image:
    """Capture the full primary monitor and return a PIL Image."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # change index if you want another display
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
        

        

def main():
    ml_comp = ML_ComputerVision()
    cont = Controller()
    checkpoint = torch.load('state_action_model.pth' ,weights_only=False)
    
    # Extract model configuration
    input_size = checkpoint['config']['input_size']
    hidden_sizes = checkpoint['config']['hidden_sizes']
    num_classes = checkpoint['config']['num_classes']
    
    model = SimpleMLP(input_size, hidden_sizes, num_classes)
    scaler = checkpoint['scaler']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(">>> Press 'P' to capture & send <Esc>; press 'Q' to quit.")
    
    try:
        while True:
            # Quit if 'q' is pressed
            if keyboard.is_pressed('q'):
                print("Quit signal received. Exiting.")
                break
            
            if keyboard.is_pressed('l'):
                print("Pause signal received. Pausing game.")
                cont.resume_game()
                time.sleep(0.1)
            if keyboard.is_pressed('k'):
                print("Pause signal received. Pausing game.")
                cont.pause_game()
                time.sleep(0.1)
                

            # Capture on 'p'
            if keyboard.is_pressed('p'):
                # Debounce so you don't trigger dozens of times per second
                # time.sleep(0.1)
                print("key pressed")
                
                img = capture_screen()
                cont.pause_game()
                
                frame_tracks , _ , _ = ml_comp.run([(img, 0)])
                print("Frame tracks:")
                state = get_state_from_positions(frame_tracks)
                action , prob = predict_action(state,model,scaler)
                time.sleep(0.1)
                print(f"Predicted action: {action} with ")
                cont.resume_game()
                cont.make_action(action)
                print(f"Action taken: {action} with ")
                
                
                
                
                
                
                # ts = int(time.time() * 1000)
                # fname = f"controller/actions/screenshot_{ts}.png"
                # print(f"[{ts}] Captured → {fname}")
                
                
                
                # Wait a bit to avoid retriggering on the same press
                # time.sleep(0.5)

            # Small sleep so this loop isn't hammering the CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user. Bye!")

if __name__ == "__main__":
    main()
