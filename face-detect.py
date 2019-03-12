#  Created by od3ng on 12/03/2019 01:17:43 PM.
#  Project: face-detection
#  File: face-detect.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import os

import cv2
from tqdm import tqdm

harr_path = 'haarcascade_frontalface_default.xml'
dataset = "dataset/source"
dataset_grayscale = "dataset/grayscale"

# create the haar cascade
faceCascade = cv2.CascadeClassifier(harr_path)

for dir_name in sorted(os.listdir(dataset)):
    path = os.path.join(dataset, dir_name)
    path_result = os.path.join(dataset_grayscale, dir_name)
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)

    for img in tqdm(os.listdir(path)):
        try:
            image = cv2.imread(os.path.join(path, img))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30)
            )
            print("Found {0} faces!".format(len(faces)))
            if len(faces) > 0:
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    crop = gray.copy()[y:y + h, x:x + w]
                    cv2.imwrite(os.path.join(path_result, img), crop)
        except Exception as e:
            pass
