"""
  face-detection
  
  Copyright (c) 2018
  All rights reserved.
  Written by od3ng created on 22/11/18 04.42
  Email   : lepengdados@gmail.com
  Github  : 0d3ng
  Hp      : 085878554150
 """

import cv2
from datetime import datetime
import time

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
captured = 0

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret)
    if frame is None:
        print("No Frame")
        continue
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} faces! {1}".format(len(faces), captured))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    if len(faces) > 0:
        crop = gray.copy()[y:y + h, x:x + w]
        cv2.imwrite("dataset/" + datetime.now().strftime('%Y%m%d%H%M%S%f') + ".jpg", crop)
        captured = captured + 1
        # time.sleep(0.25)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    time.sleep(0.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
