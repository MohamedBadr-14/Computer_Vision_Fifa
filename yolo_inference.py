from ultralytics import YOLO 

model = YOLO('models/best_initial.pt')

results = model.predict('C:/Users/moham/Downloads/testat/badr.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)