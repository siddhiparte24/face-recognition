import cv2 as cv
import numpy as np
import datetime
import csv

people = ['Elon Musk', 'Jeff Bezos', 'Kamala Harris', 'Leonardo DiCaprio', 'Narendra Modi']
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')

def markAttendance(name):
    with open('C:/Users/msi/Python/opencv/Attendance.csv', 'r+') as f:
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
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

haar = cv.CascadeClassifier('C:/Users/msi/Python/opencv/faceDetect.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera!')
    exit()
while True:
    ret, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faceD = cv.CascadeClassifier('C:/Users/msi/Python/opencv/faceDetect.xml')
    detect = faceD.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    namelist =[]

    for (x,y,w,h) in detect:
        roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        if confidence > 80:
            name = str(people[label])
            cv.putText(frame, name, (20,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
            namelist.append(name)
        elif namelist.count(name) > 5:
            cv.putText(frame, name, (20,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
        else:
            notname = str("Recognizing...")
            cv.putText(frame, notname, (20,30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

    cv.imshow("Detected", frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
markAttendance(name)

cap.release()
cv.destroyAllWindows()


