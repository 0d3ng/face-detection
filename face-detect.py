"""
  face-detection
  
  Copyright (c) 2018
  All rights reserved.
  Written by od3ng created on 21/11/18 21.47
  Email   : lepengdados@gmail.com
  Github  : 0d3ng
  Hp      : 085878554150
 """

import cv2

input_image = 'sumbing.jpeg'
harr_path = 'haarcascade_frontalface_default.xml'

# create the haara cascade
faceCascade = cv2.CascadeClassifier(harr_path)

image = cv2.imread(input_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
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
