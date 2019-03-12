#  Created by od3ng on 12/03/2019 01:17:43 PM.
#  Project: face-detection
#  File: face-detect-single.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import cv2

input_image = 'dataset/source/alan_grant/'
harr_path = 'haarcascade_frontalface_default.xml'

# create the haara cascade
faceCascade = cv2.CascadeClassifier(harr_path)

image = cv2.imread(input_image + "00000000.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))
print(faces)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
