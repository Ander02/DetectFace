import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    #For each face detected
    for (x, y, width, heigth) in faces:
    	roi_gray = gray[y:y+heigth, x:x+width] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+heigth, x:x+width]

    	#Add Label
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	color = (255, 255, 255) 
    	stroke = 1
    	cv2.putText(frame, "Anderson", (x,y), font, 1, color, stroke, cv2.LINE_AA)

    	#Save Img 
    	img_item = "imgs/" + str(int(time.time())) + ".png"
    	cv2.imwrite(img_item, roi_color)

    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + width
    	end_cord_y = y + heigth
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
    	break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
