from ultralytics import YOLO

model=YOLO("yolo11n_100_640_actuall-20250818T092347Z-1-001/yolo11n_100_640_actuall/weights/best.pt")

result=model.predict('input_videos/test.mp4',save=True)
print(result[0])

print('=====================================')
for box in result[0].boxes:
    print(box)