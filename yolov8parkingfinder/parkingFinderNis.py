import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Učitavanje slike
image_path = 'data\images\parkingNis.jpg' 
frame = cv2.imread(image_path)

# Detekcija objekata
results = model.predict(frame)
boxes = results[0].boxes.data
px = pd.DataFrame(boxes).astype("float")

print(px)

ukupanBrMesta = 15

# Prikazivanje rezultata
cv2.imshow("RGB", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
