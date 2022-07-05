#  Created by od3ng on 12/03/2019 03:38:43 PM.
#  Project: face-detection
#  File: face-detect-live.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import time
import os
import numpy as np

import cv2

output_image = 'dataset/output/nopri'
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
captured = 0
SIZE_IMG = 160
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
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces! {1}".format(len(faces), captured))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if len(faces) > 0:
            areas = [w * h for x, y, w, h in faces]
            idx = np.argmax(areas)
            biggest = faces[idx]
            # print("{} {} {} {}".format(biggest, type(biggest), biggest[0], biggest[3]))
            crop = frame.copy()[biggest[1]:biggest[1] + biggest[3], biggest[0]:biggest[0] + biggest[2]]
            captured = captured + 1
            name_file = "{}".format(f'{captured:06}')
            crop = cv2.resize(crop, (SIZE_IMG, SIZE_IMG), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(output_image, name_file + ".jpg"), crop)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
