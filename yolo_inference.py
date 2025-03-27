from ultralytics import YOLO 

model = YOLO('models/best_initial.pt')

results = model.predict('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/training_2.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)