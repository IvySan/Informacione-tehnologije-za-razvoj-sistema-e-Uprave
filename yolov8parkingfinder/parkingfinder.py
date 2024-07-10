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
image_path = 'data/images/parking.jpeg' 
frame = cv2.imread(image_path)

# Detekcija objekata
results = model.predict(frame)
boxes = results[0].boxes.data
px = pd.DataFrame(boxes).astype("float")

print(px)

ukupanBrMesta = 45

# Definicija zona na slici
area1 = [(20, 54), (20, 180), (72, 180), (72, 54)]
area2 = [(84, 54), (84, 180), (136, 180), (136, 54)]
area3 = [(148, 54), (148, 180), (200, 180), (200, 54)]
area4 = [(212, 54), (212, 180), (264, 180), (264, 54)]
area5 = [(281, 54), (281, 180), (333, 180), (333, 54)]
area6 = [(345, 54), (345, 180), (397, 180), (397, 54)]
area7 = [(409, 54), (409, 180), (461, 180), (461, 54)]
area8 = [(478, 54), (478, 180), (530, 180), (530, 54)]
area9 = [(542, 54), (542, 180), (600, 180), (600, 54)]
area10 = [(609, 54), (609, 180), (668, 180), (668, 54)]
area11 = [(674, 54), (674, 180), (732, 180), (732, 54)]
area12 =[(742, 54), (742, 180), (800, 180), (800, 54)]
area13 =[(810, 54), (810, 180), (868, 180), (868, 54)]
area14 =[(875, 54), (875, 180), (934, 180), (934, 54)]
area15 =[(942, 54), (942, 180), (992, 180), (992, 54)]

area16 = [(20, 368), (20, 486), (72, 486), (72, 368)]
area17 = [(84, 368), (84, 486), (136, 486), (136, 368)]
area18 = [(140, 370), (140, 486), (200, 486), (200, 370)]
area19 = [(208, 370), (208, 486), (264, 486), (264, 370)]
area20 = [(275, 370), (275, 496), (333, 496), (333, 370)]
area21 = [(339, 370), (339, 496), (397, 496), (397, 370)]
area22 = [(409, 370), (409, 496), (461, 496), (461, 370)]
area23 = [(473, 370), (473, 496), (530, 496), (530, 370)]
area24 = [(542, 370), (542, 496), (600, 496), (600, 370)]
area25 = [(609, 370), (609, 496), (668, 496), (668, 370)]
area26 = [(674, 370), (674, 496), (732, 496), (732, 370)]
area27 =[(742, 370), (742, 496), (800, 496), (800, 370)]
area28 =[(810, 370), (810, 496), (868, 496), (868, 370)]
area29 =[(875, 370), (875, 496), (934, 496), (934, 370)]
area30 =[(942, 370), (942, 496), (992, 496), (992, 370)]

area31 = [(18, 500), (18, 622), (72, 622), (72, 500)]
area32 = [(80, 500), (80, 622), (134, 622), (134, 500)]
area33 = [(140, 500), (140, 622), (194, 622), (194, 500)]
area34 = [(202, 500), (202, 622), (264, 622), (264, 500)]
area35 = [(275, 500), (275, 622), (333, 622), (333, 500)]
area36 = [(339, 500), (339, 622), (397, 622), (397, 500)]
area37 = [(409, 502), (409, 629), (461, 629), (461, 502)]
area38 = [(473, 505), (473, 629), (530, 629), (530, 505)]
area39 = [(542, 505), (542, 629), (600, 629), (600, 505)]
area40 = [(609, 505), (609, 629), (668, 629), (668, 505)]
area41 =[(674, 505), (674, 629), (732, 629), (732, 505)]
area42 =[(742, 505), (742, 629), (800, 629), (800, 505)]
area43 =[(810, 505), (810, 629), (868, 629), (868, 505)]
area44 =[(875, 505), (875, 629), (934, 629), (934, 505)]
area45 = [(942, 505), (942, 629), (992, 629), (992, 505)]


# Brojanje vozila u svakoj zoni
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
list9 = []
list10 = []
list11 = []
list12 = []
list13 = []
list14 = []
list15 = []
list16 = []
list17 = []
list18 = []
list19 = []
list20 = []
list21 = []
list22 = []
list23 = []
list24 = []
list25 = []
list26 = []
list27 = []
list28 = []
list29 = []
list30 = []
list31 = []
list32 = []
list33 = []
list34 = []
list35 = []
list36 = []
list37 = []
list38 = []
list39 = []
list40 = []
list41 = []
list42 = []
list43 = []
list44 = []
list45 = []

for index, row in px.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])
    d = int(row[5])
    c = class_list[d]
    confidence = row[4]
    class_index = int(row[5])
    class_name = class_list[class_index]
    
    print(f"Detektovana klasa: {class_name}, Konfidencija: {confidence}")
    
    if 'car' in c:
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        print(f"Centar automobila: ({cx}, {cy})")

        results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        print(f"Rezultat za zonu 1: {results1}")
        if results1 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list1.append(c)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
        if results2 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list2.append(c)
        
        results3 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False)
        if results3 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list3.append(c)
        
        results4 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False)
        if results4 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list4.append(c)
        
        results5 = cv2.pointPolygonTest(np.array(area5, np.int32), ((cx, cy)), False)
        if results5 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list5.append(c)
            
        results6 = cv2.pointPolygonTest(np.array(area6, np.int32), ((cx, cy)), False)
        if results6 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list6.append(c)
         
        results7 = cv2.pointPolygonTest(np.array(area7, np.int32), ((cx, cy)), False)
        if results7 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list7.append(c)
            
        results8 = cv2.pointPolygonTest(np.array(area8, np.int32), ((cx, cy)), False)
        if results8 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list8.append(c)   
        
        results9 = cv2.pointPolygonTest(np.array(area9, np.int32), ((cx, cy)), False)
        if results9 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list9.append(c)
            
        results10 = cv2.pointPolygonTest(np.array(area10, np.int32), ((cx, cy)), False)
        if results10 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list10.append(c)
            
        results11 = cv2.pointPolygonTest(np.array(area11, np.int32), ((cx, cy)), False)
        if results11 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list11.append(c)
        
        results12 = cv2.pointPolygonTest(np.array(area12, np.int32), ((cx, cy)), False)
        if results12 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list12.append(c)
            
        results13 = cv2.pointPolygonTest(np.array(area13, np.int32), ((cx, cy)), False)
        if results13 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list13.append(c)
       
        results14 = cv2.pointPolygonTest(np.array(area14, np.int32), ((cx, cy)), False)
        if results14 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list14.append(c)
            
        results15 = cv2.pointPolygonTest(np.array(area15, np.int32), ((cx, cy)), False)
        if results15 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list15.append(c)
            
        results16 = cv2.pointPolygonTest(np.array(area16, np.int32), ((cx, cy)), False)
        if results16 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list16.append(c)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        results17 = cv2.pointPolygonTest(np.array(area17, np.int32), ((cx, cy)), False)
        if results17 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list17.append(c)
        
        results18 = cv2.pointPolygonTest(np.array(area18, np.int32), ((cx, cy)), False)
        if results18 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list18.append(c)
        
        results19 = cv2.pointPolygonTest(np.array(area19, np.int32), ((cx, cy)), False)
        if results19 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list19.append(c)
        
        results20 = cv2.pointPolygonTest(np.array(area20, np.int32), ((cx, cy)), False)
        if results20 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list20.append(c)
            
        results21 = cv2.pointPolygonTest(np.array(area21, np.int32), ((cx, cy)), False)
        if results21 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list21.append(c)
            
        results22 = cv2.pointPolygonTest(np.array(area22, np.int32), ((cx, cy)), False)
        if results22 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list22.append(c)
        
        results23 = cv2.pointPolygonTest(np.array(area23, np.int32), ((cx, cy)), False)
        if results23 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list23.append(c)
        
        results24 = cv2.pointPolygonTest(np.array(area24, np.int32), ((cx, cy)), False)
        if results24 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list24.append(c)
        
        results25 = cv2.pointPolygonTest(np.array(area25, np.int32), ((cx, cy)), False)
        if results25 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list25.append(c)
            
        results26 = cv2.pointPolygonTest(np.array(area26, np.int32), ((cx, cy)), False)
        if results26 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list26.append(c)
         
        results27 = cv2.pointPolygonTest(np.array(area27, np.int32), ((cx, cy)), False)
        if results27 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list27.append(c)
            
        results28 = cv2.pointPolygonTest(np.array(area28, np.int32), ((cx, cy)), False)
        if results28 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list28.append(c)   
        
        results29 = cv2.pointPolygonTest(np.array(area29, np.int32), ((cx, cy)), False)
        if results29 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list29.append(c)
            
        results30 = cv2.pointPolygonTest(np.array(area30, np.int32), ((cx, cy)), False)
        if results30 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list30.append(c)
            
        results31 = cv2.pointPolygonTest(np.array(area31, np.int32), ((cx, cy)), False)
        if results31 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list31.append(c)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        results32 = cv2.pointPolygonTest(np.array(area32, np.int32), ((cx, cy)), False)
        if results32 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list32.append(c)
        
        results33 = cv2.pointPolygonTest(np.array(area33, np.int32), ((cx, cy)), False)
        if results33 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list33.append(c)
        
        results34 = cv2.pointPolygonTest(np.array(area34, np.int32), ((cx, cy)), False)
        if results34 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list34.append(c)
        
        results35 = cv2.pointPolygonTest(np.array(area35, np.int32), ((cx, cy)), False)
        if results35 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list35.append(c)
            
        results36 = cv2.pointPolygonTest(np.array(area36, np.int32), ((cx, cy)), False)
        if results36 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list36.append(c)
         
        results7 = cv2.pointPolygonTest(np.array(area7, np.int32), ((cx, cy)), False)
        if results7 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list7.append(c)
            
        results38 = cv2.pointPolygonTest(np.array(area38, np.int32), ((cx, cy)), False)
        if results38 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list38.append(c)   
        
        results39 = cv2.pointPolygonTest(np.array(area39, np.int32), ((cx, cy)), False)
        if results39 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list39.append(c)
            
        results40 = cv2.pointPolygonTest(np.array(area40, np.int32), ((cx, cy)), False)
        if results40 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list40.append(c)
            
        results41 = cv2.pointPolygonTest(np.array(area41, np.int32), ((cx, cy)), False)
        if results41 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list41.append(c)
        
        results42 = cv2.pointPolygonTest(np.array(area42, np.int32), ((cx, cy)), False)
        if results42 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list42.append(c)
            
        results43 = cv2.pointPolygonTest(np.array(area43, np.int32), ((cx, cy)), False)
        if results43 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list43.append(c)
       
        results44 = cv2.pointPolygonTest(np.array(area44, np.int32), ((cx, cy)), False)
        if results44 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list44.append(c)
            
        results45 = cv2.pointPolygonTest(np.array(area45, np.int32), ((cx, cy)), False)
        if results45 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list45.append(c)
            
        

# Vizualizacija rezultata na slici
a1 = len(list1)
a2 = len(list2)
a3 = len(list3)
a4 = len(list4)
a5 = len(list5)
a6 = len(list6)
a7 = len(list7)
a8 = len(list8)
a9 = len(list9)
a10 = len(list10)
a11 = len(list11)
a12 = len(list12)
a13 = len(list13)
a14 = len(list14)
a15 = len(list15)
a16 = len(list16)
a17 = len(list17)
a18 = len(list18)
a19 = len(list19)
a20 = len(list20)
a21 = len(list21)
a22 = len(list22)
a23 = len(list23)
a24 = len(list24)
a25 = len(list25)
a26 = len(list26)
a27 = len(list27)
a28 = len(list28)
a29 = len(list29)
a30 = len(list30)
a31 = len(list31)
a32 = len(list32)
a33 = len(list33)
a34 = len(list34)
a35 = len(list35)
a36 = len(list36)
a37 = len(list37)
a38 = len(list38)
a39 = len(list39)
a40 = len(list40)
a41 = len(list41)
a42 = len(list42)
a43 = len(list43)
a44 = len(list44)
a45 = len(list45)

print(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15)

zauzeto=(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15+a16+a17+a18+a19+a20+a21+a22+a23+a24+a25+a26+a27+a28+a29+a30+a31+a32+a33+a34+a35+a36+a37+a38+a39+a40+a41+a4+a43+a44+a45)
zauzeto = 16
slobodno=(ukupanBrMesta-zauzeto)


# Crtanje poligona i brojeva na slici za svaku zonu
if a1==1:
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2) #zauzeto
else:
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)#slobodno

if a2==1:
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
    
if a3==1:
    cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 255, 0), 2)
    
if a4==1:
    cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 255, 0), 2)
    
if a5==1:
    cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 255, 0), 2)
    
if a6==0:
    cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 255, 0), 2)
    
if a7==1:
    cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 255, 0), 2)
    
if a8==0:
    cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 255, 0), 2)

if a9==1:
    cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 255, 0), 2)
    
if a10==0:
    cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 255, 0), 2)
    
if a11==1:
    cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 255, 0), 2)
    
if a12==1:
    cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 255, 0), 2)
    
if a13==1:
    cv2.polylines(frame, [np.array(area13, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area13, np.int32)], True, (0, 255, 0), 2)
    
if a14==1:
    cv2.polylines(frame, [np.array(area14, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area14, np.int32)], True, (0, 255, 0), 2)
    
if a15==1:
    cv2.polylines(frame, [np.array(area15, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area15, np.int32)], True, (0, 255, 0), 2)
    
if a16==1:
    cv2.polylines(frame, [np.array(area16, np.int32)], True, (0, 0, 255), 2) #zauzeto
else:
    cv2.polylines(frame, [np.array(area16, np.int32)], True, (0, 255, 0), 2)#slobodno

if a17==1:
    cv2.polylines(frame, [np.array(area17, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area17, np.int32)], True, (0, 255, 0), 2)
    
if a18==0:
    cv2.polylines(frame, [np.array(area18, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area18, np.int32)], True, (0, 255, 0), 2)
    
if a19==0:
    cv2.polylines(frame, [np.array(area19, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area19, np.int32)], True, (0, 255, 0), 2)
    
if a20==1:
    cv2.polylines(frame, [np.array(area20, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area20, np.int32)], True, (0, 255, 0), 2)
    
if a21==0:
    cv2.polylines(frame, [np.array(area21, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area21, np.int32)], True, (0, 255, 0), 2)
    
if a22==0:
    cv2.polylines(frame, [np.array(area22, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area22, np.int32)], True, (0, 255, 0), 2)
    
if a23==0:
    cv2.polylines(frame, [np.array(area23, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area23, np.int32)], True, (0, 255, 0), 2)

if a24==1:
    cv2.polylines(frame, [np.array(area24, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area24, np.int32)], True, (0, 255, 0), 2)
    
if a25==1:
    cv2.polylines(frame, [np.array(area25, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area25, np.int32)], True, (0, 255, 0), 2)
    
if a26==1:
    cv2.polylines(frame, [np.array(area26, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area26, np.int32)], True, (0, 255, 0), 2)
    
if a27==0:
    cv2.polylines(frame, [np.array(area27, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area27, np.int32)], True, (0, 255, 0), 2)
    
if a28==0:
    cv2.polylines(frame, [np.array(area28, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area28, np.int32)], True, (0, 255, 0), 2)
    
if a29==1:
    cv2.polylines(frame, [np.array(area29, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area29, np.int32)], True, (0, 255, 0), 2)
    
if a30==1:
    cv2.polylines(frame, [np.array(area30, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area30, np.int32)], True, (0, 255, 0), 2)
    
if a31==1:
    cv2.polylines(frame, [np.array(area31, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area31, np.int32)], True, (0, 255, 0), 2)
    
if a32==0:
    cv2.polylines(frame, [np.array(area32, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area32, np.int32)], True, (0, 255, 0), 2)
    
if a33==0:
    cv2.polylines(frame, [np.array(area33, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area33, np.int32)], True, (0, 255, 0), 2)
    
if a34==0:
    cv2.polylines(frame, [np.array(area34, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area34, np.int32)], True, (0, 255, 0), 2)
    
if a35==1:
    cv2.polylines(frame, [np.array(area35, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area35, np.int32)], True, (0, 255, 0), 2)
    
if a36==1:
    cv2.polylines(frame, [np.array(area36, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area36, np.int32)], True, (0, 255, 0), 2)
    
if a36==1:
    cv2.polylines(frame, [np.array(area36, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area36, np.int32)], True, (0, 255, 0), 2)
    
if a37==1:
    cv2.polylines(frame, [np.array(area37, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area37, np.int32)], True, (0, 255, 0), 2)
    
if a38==1:
    cv2.polylines(frame, [np.array(area38, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area38, np.int32)], True, (0, 255, 0), 2)
    
if a39==0:
    cv2.polylines(frame, [np.array(area39, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area39, np.int32)], True, (0, 255, 0), 2)
    
if a40==1:
    cv2.polylines(frame, [np.array(area40, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area40, np.int32)], True, (0, 255, 0), 2)
    
if a41==0:
    cv2.polylines(frame, [np.array(area41, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area41, np.int32)], True, (0, 255, 0), 2)
    
if a42==1:
    cv2.polylines(frame, [np.array(area42, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area42, np.int32)], True, (0, 255, 0), 2)
    
if a43==0:
    cv2.polylines(frame, [np.array(area43, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area43, np.int32)], True, (0, 255, 0), 2)
    
if a44==1:
    cv2.polylines(frame, [np.array(area44, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area44, np.int32)], True, (0, 255, 0), 2)
    
if a45==1:
    cv2.polylines(frame, [np.array(area45, np.int32)], True, (0, 0, 255), 2) 
else:
    cv2.polylines(frame, [np.array(area45, np.int32)], True, (0, 255, 0), 2)
    


cv2.putText(frame,"Broj slobodnih mesta je:" + str(slobodno), (285, 238), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(frame,"Broj zauzetih mesta je:" + str(zauzeto), (290, 264), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1)

# Prikazivanje rezultata
cv2.imshow("RGB", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()