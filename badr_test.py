from computer_vision import ComputerVision



comp = ComputerVision()
frames = comp.read_frames("./badr/all")
comp.run(frames, attempts=1)