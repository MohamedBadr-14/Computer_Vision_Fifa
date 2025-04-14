from ultralytics import YOLO 

model = YOLO('models/best_new.pt')

results = model.predict('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/ocr_extractor/test_5.png',save=True)
#print(results[0])
#print('=====================================')
for box in results[0].boxes:
    print(box)
