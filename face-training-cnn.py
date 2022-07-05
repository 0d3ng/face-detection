#  Created by od3ng on 05/07/2022 12:31:35 AM.
#  Project: face-detection
#  File: face-training-cnn.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#

import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Input
# keras imports
from keras.models import Model
# from tensorflow.keras import layers, Model, Sequential, Input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

DATA_SET = "dataset/output"
# DATA_SET_TR = "dataset/output_test/train"
# DATA_SET_VAL = "dataset/output_test/val"

image_size = 160
batch_size = 32
classes = 6
epochs = 5
lr = 0.0001

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATA_SET,
#     validation_split=0.2,
#     subset="training",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATA_SET,
#     validation_split=0.2,
#     subset="validation",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")
# plt.show()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # set validation split

# print("type of {}".format(type(train_datagen)))

train_generator = train_datagen.flow_from_directory(
    DATA_SET,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    subset='training')  # set as training data

# print("type of {}".format(type(train_generator)))

validation_generator = train_datagen.flow_from_directory(
    DATA_SET,  # same directory as training data
    target_size=(image_size, image_size),
    batch_size=batch_size,
    subset='validation')  # set as validation data
# print("type of {}".format(type(validation_generator)))

# Rescale Testing Data
test_datagen = ImageDataGenerator(rescale=1. / 255)

base = tf.keras.applications.MobileNetV2(
    include_top=False,
    alpha=0.35,
    weights='imagenet',
    input_tensor=Input(shape=(image_size, image_size, 3)),
    input_shape=(image_size, image_size, 3))

base.trainable = False

base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                               input_tensor=Input(shape=(image_size, image_size, 3)),
                                               input_shape=(image_size, image_size, 3))
top_layers = base_model.output
top_layers = GlobalAveragePooling2D()(top_layers)
top_layers = Dense(1024, activation='relu')(top_layers)
predictions = Dense(classes, activation='softmax')(top_layers)
model = Model(inputs=base_model.input, outputs=predictions)

# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

model.save("save_model.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
