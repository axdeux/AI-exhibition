import cv2
import numpy as np
import os
import sys
import detection
from deepface import DeepFace




# initializing face recognition methods
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(detection.model_file)
person_id2name = detection.read_person_names(detection.person_names_file)
undefined_person = 'Unknown'
confidence_threshold = 20
count1 = 50
face_analying_time = 50
people = dict()
for names in person_id2name.values():
    people[names] = ['unknown', 'unknown', 'unknown']
people[undefined_person] = ['unknown', 'unknown', 'unknown']




# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
font = cv2.FONT_HERSHEY_SIMPLEX
count2 = 0
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # mirror
    faces,gray = detection.get_faces(img, face_detector)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        person_id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if ((100 - confidence) > confidence_threshold):
            person_name = person_id2name[str(person_id)]
            confidence = "  {0}%".format(round(100 - confidence))
            count1 += 1
            if count1 > face_analying_time:
                estimation = DeepFace.analyze(img, actions = ['age', 'gender', 'emotion'], enforce_detection=False)
                count2 +=1
                print(estimation, count2)
                people[person_name] = [estimation['age'], estimation['gender'],estimation['dominant_emotion']]
                count1 = 0
            else:
                pass
        else:
            person_name = undefined_person
            confidence = "  {0}%".format(round(100 - confidence))
        if isinstance(estimation, dict):
            x1, y1, w1, h1 = map(int, estimation['region'].values())
            cv2.rectangle(img, (x1,y1), (x1+w1,y1+h1), (0, 0, 255), 2)
        cv2.putText(img, 'Age: '+ str(people[person_name][0]), (x+w,y+15), font, 0.5, (0,0,0), 2)
        cv2.putText(img, 'Gender: '+ str(people[person_name][1]), (x+w,y+30), font, 0.5, (0,0,0), 2)
        cv2.putText(img, 'Emotion: '+ str(people[person_name][2]), (x+w,y+45), font, 0.5, (0,0, 0), 2)
        cv2.putText(img, str(person_name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    cv2.imshow('camera',img)
    k = cv2.waitKey(5) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()