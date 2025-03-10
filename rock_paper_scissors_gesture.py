# -*- coding: utf-8 -*-
"""rock paper scissors gesture.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gHTnbv-mioHQwyDBwtWLJj0FDL4uvUCc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

### identify the gesture of the rock paper and scissor gesture

import kagglehub

# Download latest version
path = kagglehub.dataset_download("josiagiven/neural-network")

print("Path to dataset files:", path)

import shutil

shutil.move(path, "/content")

path = "/content/1/suit/images/"
path_train = path+"train/"
path_val = path+"val/"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(
 rescale = 1./255,
 rotation_range=40,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 fill_mode='nearest'
)

train_generator = training_datagen.flow_from_directory(
 path_train,
 target_size=(150,150),
 class_mode='categorical'
)

val_generator = training_datagen.flow_from_directory(
 path_val,
 target_size=(150,150),
 class_mode='categorical'
)

model = tf.keras.models.Sequential([
 # Note the input shape is the desired size of the image:
 # 150x150 with 3 bytes color
 # This is the first convolution
 tf.keras.layers.Conv2D(64, (3,3), activation='relu',
 input_shape=(150, 150, 3)),
 tf.keras.layers.MaxPooling2D(2, 2),
 # The second convolution
 tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
 tf.keras.layers.MaxPooling2D(2,2),
 # The third convolution
 tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
 tf.keras.layers.MaxPooling2D(2,2),
 # The fourth convolution
 tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
 tf.keras.layers.MaxPooling2D(2,2),
 # Flatten the results to feed into a DNN
 tf.keras.layers.Flatten(),
 # 512 neuron hidden layer
 tf.keras.layers.Dense(512, activation='relu'),
 tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',
 metrics=['accuracy'])

history = model.fit(train_generator, epochs=5,
 validation_data = val_generator, verbose = 1)

from google.colab import files
from keras.preprocessing import image
uploaded = files.upload()
for fn in uploaded.keys():
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)

# Save as .h5 file
model.save("/content/rps_model.h5")

# Create a directory to upload
import os
os.makedirs("/content/rps_model_dir", exist_ok=True)
os.rename("/content/rps_model.h5", "/content/rps_model_dir/rps_model.h5")

from huggingface_hub import login
login(token="")

from tensorflow.keras.models import save_model
from huggingface_hub import HfApi, HfFolder

# Upload the model manually
from huggingface_hub import HfApi

api = HfApi()

try:
  repo_url = api.create_repo(repo_id="navidfalah/Rock-paper-scissors-model", private=False)
except:
  pass

api.upload_folder(
    folder_path="/content/rps_model_dir",
    repo_id="navidfalah/Rock-paper-scissors-model"
)

