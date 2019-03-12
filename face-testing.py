#  Created by od3ng on 12/03/2019 02:07:10 PM.
#  Project: face-detection
#  File: face-testing.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

#  Created by od3ng on 12/03/2019 01:19:17 PM.
#  Project: face-detection
#  File: face-testing.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import os
import time

import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("face.model")

data_dir = "dataset/grayscale"
dirs = []

for dir_name in sorted(os.listdir(data_dir)):
    dirs.append(dir_name)

width = 100
height = 100

while True:
    ret, frame = cap.read()
    if frame is None:
        print("No frame")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(faces) > 0:
        crop = gray.copy()[y:y + h, x:x + w]
        img_array = cv2.resize(crop.copy(), (width, height))
        new_array = np.array(img_array).reshape(-1, width, height, 1)
        new_array = new_array / 255.0

        prediction = model.predict(new_array)
        idx = np.argmax(prediction[0])
        print("index {}, name is {}".format(idx, dirs[idx]))
        cv2.putText(frame, dirs[idx], (x + int(w / 2) - 10, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("face detection", frame)
    time.sleep(0.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
