#  Created by od3ng on 12/03/2019 03:38:43 PM.
#  Project: face-detection
#  File: face-testing.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import os
import time

import cv2
import numpy as np
from keras.applications.vgg19 import preprocess_input
from keras.models import load_model
from tensorflow.keras.preprocessing import image

cap = cv2.VideoCapture(0)

start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("save_model.h5")
cv2.namedWindow("face testing", cv2.WINDOW_GUI_EXPANDED)

data_dir = "dataset/output"
dirs = []

for dir_name in sorted(os.listdir(data_dir)):
    dirs.append(dir_name)

width, height = 160, 160

while True:
    ret, frame = cap.read()

    fc += 1
    TIME = time.time() - start_time

    if TIME >= display_time:
        FPS = fc / TIME
        fc = 0
        start_time = time.time()

    fps_disp = "FPS: " + "{:.2f}".format(FPS)

    if frame is None:
        print("No frame")
        continue

        # Add FPS count on frame
    cv2.putText(frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if len(faces) > 0:
            new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = new_frame.copy()[y:y + h, x:x + w]
            img_array = cv2.resize(crop.copy(), (width, height))
            x_img = image.img_to_array(img_array)
            x_img = np.expand_dims(x_img, axis=0)
            x_img = preprocess_input(x_img)
            y_prob = model.predict(x_img)
            y_class = y_prob.argmax(axis=-1)
            y_class = y_class[0]
            y_confidence = y_prob[0][y_class] * 100
            print("predicted label: {} (prob = {})".format(y_class, y_confidence))
            print("index {}, name is {}".format(y_class, dirs[y_class]))
            text = "{} {}%".format(dirs[y_class], "{:.2f}".format(y_confidence))
            print("text {}".format(text))
            cv2.putText(frame, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("face testing", frame)
    time.sleep(0.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
