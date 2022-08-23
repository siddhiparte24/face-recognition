import os
import cv2 as cv
import numpy as np

people = ['Elon Musk', 'Jeff Bezos', 'Kamala Harris', 'Leonardo DiCaprio', 'Narendra Modi']

DIR = r'C:/Users/msi/Python/opencv/faces'

haar = cv.CascadeClassifier('faceDetect.xml')

features = []
labels = []

# first commit
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            imgPath = os.path.join(path, img)

            imgArray = cv.imread(imgPath)
            gray = cv.cvtColor(imgArray, cv.COLOR_BGR2GRAY)

            detect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
            for (x,y,w,h) in detect:
                roi = gray[y:y+h, x:x+w]
                features.append(roi)
                labels.append(label)

create_train()

print('training done----------------------')




features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('label.npy', labels)
        


