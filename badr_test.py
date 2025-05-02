from computer_vision import ComputerVision
from utils import read_video, save_video



comp = ComputerVision()
frames = comp.read_frames("D:/Computer_Vision_Fifa/frames")
comp.run(frames, attempts=1)