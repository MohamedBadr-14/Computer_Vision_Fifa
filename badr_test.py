from computer_vision import ComputerVision
import cv2



comp = ComputerVision()
frames = comp.read_frames("./badr/all")
# frames = [cv2.imread("./input_videos/test_offside.png")]
comp.run(frames, attempts=1)