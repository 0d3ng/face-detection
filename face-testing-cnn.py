#  Created by od3ng on 12/03/2019 03:38:43 PM.
#  Project: face-detection
#  File: face-testing.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import time

import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

cap = cv2.VideoCapture(0)

start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# model = load_model("save_model.h5")
model = load_model("bahit_model_v2.h5")
cv2.namedWindow("face testing", cv2.WINDOW_GUI_EXPANDED)


class_names = ['alvin', 'amy', 'anti', 'bahit', 'farida', 'hafizi']

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
            # Convert the captured frame into RGB
            crop = frame.copy()[y:y + h, x:x + w]
            im = Image.fromarray(frame, 'RGB')
            # Resizing into dimensions you used while training
            im = im.resize((width, height))
            img_array = np.array(im)

            # Expand dimensions to match the 4D Tensor shape.
            img_array = np.expand_dims(img_array, axis=0)

            # Calling the predict function using keras
            prediction = model.predict(img_array)  # [0][0]
            print(prediction)

            y_class = prediction.argmax(axis=-1)
            y_class = y_class[0]
            y_confidence = prediction[0][y_class] * 100
            print("predicted label: {} (prob = {})".format(y_class, y_confidence))
            print("index {}, name is {}".format(y_class, class_names[y_class]))
            text = "{} {}%".format(class_names[y_class], "{:.2f}".format(y_confidence))
            print("text {}".format(text))
            cv2.putText(frame, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("face testing", frame)
    time.sleep(0.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
