import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import pytz

# Initialize Firebase
# Set the FIREBASE_CRED_PATH environment variable to point to your service
# account JSON key file. This keeps the key out of source control.
# e.g. on Linux/Mac: export FIREBASE_CRED_PATH=/path/to/your-key.json
cred_path = os.environ.get('FIREBASE_CRED_PATH')
if not cred_path:
    raise RuntimeError(
        "FIREBASE_CRED_PATH environment variable is not set. "
        "Point it to your Firebase service account JSON key file."
    )
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Add to Firebase Firestore
model = YOLO('yolov8n.pt')

#area1 = [(440, 500), (460, 500), (460, 2), (440, 2)]
#area2 = [(465, 500), (485, 500), (485, 2), (465, 2)]
area1 = [(440, 500), (640, 500), (640, 100), (440, 100)]
area2 = [(655, 500), (855, 500), (855, 100), (655, 100)]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#cap = cv2.VideoCapture('testedit.mp4')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (1020, 500), isColor=False)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
i = 0
j = 0
people_entering = {}
entering = set()
people_exiting = {}
exiting = set()
tracker = Tracker()
fps = 10

# Define the Irish timezone
ireland_tz = pytz.timezone('Europe/Dublin')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox

        results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results >= 0:
            print(f"r1 running")
            if id not in people_entering:
                people_entering[id] = count
                print(f"Added ID {people_entering[id]} to people_entering")
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)
            i += 1

        if id in people_entering:
            print(f"r2 running")
            results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            print(results1)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 1)
                cv2.circle(frame, (x4, y4), 2, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                entering.add(id)
                print(f"ID {id} entered area2")

        results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results2 >= 0:
            print(f"r3 running")
            if id not in people_exiting:
                people_exiting[id] = count
                print(f"Added ID {people_exiting[id]} to people_exiting")
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            j += 1
        if id in people_exiting:
            print(f"r4 running")
            results3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 2, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                print(f"ID {id} re-entered area1 from area2")

        len_entering = len(people_entering)
        len_exiting = len(people_exiting)
        if (results >= 0 or results2 >= 0) :
            now_utc = datetime.datetime.utcnow()
            now_ist = now_utc.replace(tzinfo=pytz.utc).astimezone(ireland_tz)
            now_str = now_ist.strftime('%Y-%m-%d %H:%M:%S')

            if people_entering.get(id, 0) < people_exiting.get(id, 0):
                entering.add(id)
                entering_ref = db.collection('entering')
                entering_ref.add({'person_id': len(entering), 'time': now_str})
                people = len(entering) - len(exiting)
                people_ref = db.collection('people')
                people_ref.add({'person_id': people, 'time': now_str})

            if people_entering.get(id, 0) > people_exiting.get(id, 0):
                exiting.add(id)
                exiting_ref = db.collection('exiting')
                exiting_ref.add({'person_id': len(exiting), 'time': now_str})

            people = len(entering) - len(exiting)
            people_ref = db.collection('people')
            people_ref.add({'person_id': people, 'time': now_str})

    print(f"People entering: {people_entering}")
    print(f"People exiting: {people_exiting}")

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    print(f"Number of entering IDs: {len(entering)}")
    print(f"Number of exiting IDs: {len(exiting)}")
    cv2.imshow("Grayscale", frame)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

if os.path.exists(video_path):
    os.remove(video_path)
    print(f"Deleted file: {video_path}")
else:
    print(f"File not found: {video_path}")

