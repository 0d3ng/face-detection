#  Created by od3ng on 12/03/2019 03:38:43 PM.
#  Project: face-detection
#  File: face-detect-live.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import time
from datetime import datetime

import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
captured = 0
cv2.namedWindow("face grabber", cv2.WINDOW_GUI_EXPANDED)

while True:
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if len(faces) > 0:
            crop = gray.copy()[y:y + h, x:x + w]
            cv2.imwrite("dataset/" + datetime.now().strftime('%Y%m%d%H%M%S%f') + ".jpg", crop)
            captured = captured + 1
            cv2.putText(frame, "Face found!".format(captured), (x, y - 30),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)
            cv2.putText(frame, "Captured {} times".format(captured), (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)
            # time.sleep(0.25)
    # Display the resulting frame
    cv2.imshow('face grabber', frame)
    time.sleep(0.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
