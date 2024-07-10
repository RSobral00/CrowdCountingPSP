


import time
import os
import random

import tensorflow as tf
import cv2
import tensorflow_addons as tfa
import scipy.io as io
import numpy as np
import sys

import keras
from keras import backend as K
from keras.models import load_model

def euclidean_distance_loss(y_true, y_pred):
    return K.square(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

# Check Available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Name GPUs Available: ", tf.config.experimental.list_physical_devices("GPU"))
print("GPU_INFO", tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

input_folder = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/test_images")
output_folder = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/PSP_DC_Soft_CSRNETP_Predictions")

model_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingUAV/Results/PSP_DC_Soft_CSRNETP")
loaded_model = tf.keras.models.load_model(model_path, compile=True, custom_objects={"euclidean_distance_loss": euclidean_distance_loss})



image_files = [f for f in os.listdir(input_folder) if f.endswith('.JPG')]
print(image_files)

# Measure the prediction time separately
total_predict_time = 0

for image_file in image_files:
    print(image_file)
    image_path = os.path.join(input_folder, image_file)
    
    img = [cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float16) / 255]
    img = np.array(img)
    print(img.shape)
    
    # Measure prediction time
    predict_start_time = time.time()
    img_predict = loaded_model.predict(img)
    predict_end_time = time.time()
    total_predict_time += (predict_end_time - predict_start_time)
    
    img_predict = img_predict * 0.042
    output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_prediction.mat")
    io.savemat(output_path, {"heatmap": img_predict.astype(np.float16)})
    
    print(f"Image Title: {image_file}")
    print(f"Prediction Sum: {np.sum(img_predict)}")
    print()

num_images = len(image_files)
predict_fps = num_images / total_predict_time

print(f"Processed {num_images} images")
print(f"Prediction FPS (Frames per Second): {predict_fps:.2f}")

