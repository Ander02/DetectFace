import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, videoImg = cap.read()
    
    gray = cv2.cvtColor(videoImg, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)    
    
    for (x,y,width,heigth) in faces:
        cv2.rectangle(videoImg,(x,y),(x+width,y+heigth),(255,0,0),2)
        
        roi_gray = gray[y:y+heigth, x:x+width]

    cv2.imshow('img',videoImg)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()