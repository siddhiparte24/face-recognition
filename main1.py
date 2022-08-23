import imp
import cv2 as cv
import numpy as np
from datetime import datetime

people = ['Elon Musk', 'Jeff Bezos', 'Kamala Harris', 'Leonardo DiCaprio', 'Narendra Modi']
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')

haar = cv.CascadeClassifier('faceDetect.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


# Edited By Shreyan 
# Function for Marking the Attendance

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        # Now we read all the lines in our data
        # If somebody is already arrived we don't want to repeat it
        myDataList = f.readlines()
        nameList = []
        # We want to put all the names we find in this list
        for line in myDataList:
            entry = line.split(',')
            # We want to split the list in name and time
            nameList.append(entry[0])
        # print(nameList)
        # Entry 0 will be the names
        # if name is not present
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera!')
    exit()
while True:
    ret, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faceD = cv.CascadeClassifier('faceDetect.xml')
    detect = faceD.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    for (x,y,w,h) in detect:
        roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        cv.putText(frame, str(people[label]), (20,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=1)


        # Make changes here
        # Done By Shreyan
        name = str(people[label])
        # markAttendance(name)

    cv.imshow("Detected", frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()


