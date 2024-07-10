# Rúben Sobral - Model Testing on DroneCrowd


import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import scipy.io as io 
import numpy as np
import pandas as pd



from keras.layers import BatchNormalization

from sklearn.metrics import mean_squared_error
from keras.initializers import RandomNormal
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.models import Model,Sequential
from keras.layers import *
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io
import os
import glob
import cv2
import random
import math
import sys
import numpy as np
import h5py
import json
import chardet

import time
import os
import random
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'


import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import scipy.io as io 
import numpy as np


import keras
from keras import backend as K
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D,
    DepthwiseConv2D, GlobalMaxPooling2D, Multiply, Add, Reshape,
    Concatenate, UpSampling2D, Cropping2D, BatchNormalization, ReLU,Activation,Lambda,Dropout,Flatten
)
from keras.utils import Sequence
from keras.callbacks import Callback
from keras.models import load_model
########################### RUN WITH GPU ############################################
#Check Available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Name GPUs Available: ", tf.config.experimental.list_physical_devices("GPU"))


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate memory on the second GPU
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


#Load desired model to be tested.
model_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/ARCN_Model")


def euclidean_distance_loss(y_true, y_pred):

    return K.square(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


loaded_model = tf.keras.models.load_model(model_path, compile=True, custom_objects={'euclidean_distance_loss': euclidean_distance_loss})




test_info = os.path.expanduser("~/Desktop/Tese/DroneCrowd/test_data/test_info.txt")

test_image_folder = "~/Desktop/Tese/DroneCrowd/test_data/images_test/"

test_info = pd.read_csv(test_info,sep=",",header=None)

test_info.columns=["Img","N_Gt_Int","N_Gt"]
 
test_info["Img"] = test_info["Img"].apply(lambda x: "0" + str(x) if len(str(x)) == 5 else str(x))

test_info["Img"] = "img" + test_info["Img"] + ".jpg"





def process_image(value):
    img_path = os.path.expanduser(os.path.join(test_image_folder, value))
    
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)/255
    img = np.expand_dims(img, axis=0)
    
    start_time = time.time()  
    
    predicted_label = loaded_model.predict(img)
    
    predicted_label *= 0.042  
    
    end_time = time.time()  
    
    return np.sum(predicted_label), end_time - start_time  


overall_fps = 0
for idx, img_name in enumerate(test_info["Img"]):
    prediction, time_taken = process_image(img_name)
    overall_fps += 1 / time_taken  
    test_info.at[idx, "N_Pred"] = prediction  

overall_fps /= len(test_info)  


test_info.to_csv(os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/Test_Info_ARCN_Model.csv"), index=False)

print("Overall FPS:", overall_fps)
