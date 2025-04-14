from ultralytics import YOLO 

model = YOLO('models/keypoints.pt')

results = model.predict('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/',save=True)
#print(results[0])
#print('=====================================')
for box in results[0].boxes:
    print(box)
