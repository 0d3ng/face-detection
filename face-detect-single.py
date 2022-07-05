#  Created by od3ng on 12/03/2019 01:17:43 PM.
#  Project: face-detection
#  File: face-detect-single.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import cv2
import os
import numpy as np

input_image = 'dataset/new/madonna'
output_image = 'dataset/output/madonna'
harr_path = 'haarcascade_frontalface_default.xml'
# create the haara cascade
faceCascade = cv2.CascadeClassifier(harr_path)
captured: int = 0
SIZE_IMG = 160

dir = os.listdir(input_image)
for file in dir:
    print(file)
    path_img = os.path.join(input_image, file)
    image = cv2.imread(path_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    # print(faces)
    if len(faces) > 0:
        areas = [w * h for x, y, w, h in faces]
        idx = np.argmax(areas)
        biggest = faces[idx]
        # print("{} {} {} {}".format(biggest, type(biggest), biggest[0], biggest[3]))
        crop = image.copy()[biggest[1]:biggest[1] + biggest[3], biggest[0]:biggest[0] + biggest[2]]
        captured = captured + 1
        name_file = "{}".format(f'{captured:06}')
        # print(name_file)
        # print(crop)
        crop = cv2.resize(crop, (SIZE_IMG, SIZE_IMG), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_image, name_file + ".jpg"), crop)

    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
