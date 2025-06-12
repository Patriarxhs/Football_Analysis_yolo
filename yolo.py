from ultralytics import YOLO

model=YOLO("models/best_100_640.pt")

result=model.predict('input_videos/background-10.mp4',save=True,stream=True)
print(result[0])

print('=====================================')
for box in result[0].boxes:
    print(box)