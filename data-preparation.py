#  Created by od3ng on 12/03/2019 01:17:43 PM.
#  Project: face-detection
#  File: data-preparation.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import os
import pickle
import random

import cv2
import numpy as np
from tqdm import tqdm

DATADIR = "dataset/grayscale"
dirs = []

training_data = []
width, height = 100, 100

for dir_name in sorted(os.listdir(DATADIR)):
    dirs.append(dir_name)

for dir_name in dirs:
    path = os.path.join(DATADIR, dir_name)
    class_num = dirs.index(dir_name)
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (width, height))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, width, height, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
