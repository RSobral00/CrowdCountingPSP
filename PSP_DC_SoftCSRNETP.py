# Fine tuning da SoftCSRNet+ no conjunto de dados da PSP

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
# Standard Libraries
import time
import os
import random

#Turn oneDNN custom operations off
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# External Libraries
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import scipy.io as io 
import numpy as np

# Keras and TensorFlow

from keras import backend as K
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D,
    DepthwiseConv2D, GlobalMaxPooling2D, Multiply, Add, Reshape,
    Concatenate, UpSampling2D, Cropping2D, BatchNormalization, ReLU,Activation,Lambda,Dropout,Flatten
)
from keras.utils import Sequence
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

########################### RUN WITH GPU ############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
       
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        
        print(e)

#######################################################################################




# Set random seed for Python environment
seed = 42
random.seed(seed)
np.random.seed(seed)

# Set random seed for TensorFlow
tf.random.set_seed(seed)




#Define file paths
train_folder_features = "~/Desktop/Tese/PSP_Dataset/train_images"
train_folder_labels = "~/Desktop/Tese/PSP_Dataset/train_labels"
val_folder_features = "~/Desktop/Tese/PSP_Dataset/val_images"         
val_folder_labels = "~/Desktop/Tese/PSP_Dataset/val_labels"


# List file names and shuffle them
names_train = os.listdir(os.path.expanduser(train_folder_labels))
names_val = os.listdir(os.path.expanduser(val_folder_labels))

names_train = [os.path.splitext(file_name)[0] for file_name in names_train]
names_val = [os.path.splitext(file_name)[0] for file_name in names_val]





random.shuffle(names_train )
random.shuffle(names_val)

print("---names_train generation completed---")
print("---names_val generation completed---")


model_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/Soft_CSRNETP")


# Load the pre-trained CSRNet model

def euclidean_distance_loss(y_true, y_pred):

    return K.square(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


pretrained_model = tf.keras.models.load_model(model_path, custom_objects={'euclidean_distance_loss': euclidean_distance_loss})


for layer in pretrained_model.layers:
    layer.trainable = False


for layer in pretrained_model.layers[-9:]:
    layer.trainable = True


pretrained_model.summary()



early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

opt = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.95)




EPOCHS = 300

pretrained_model.compile(optimizer=opt,   
              loss=euclidean_distance_loss,  
              metrics=["mse", "mae"])  


filename = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/PSP_DC_SoftCSRNETP_history.csv")
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)




class CustomDataGenerator(Sequence):
    def __init__(self, folder_features, folder_labels, names, batch_size, flip_probability=0.5):
        self.folder_features = folder_features
        self.folder_labels = folder_labels
        self.names = names
        self.batch_size = batch_size
        self.num_samples = len(names)
        self.on_epoch_end()
        self.flip_probability = flip_probability  

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_names = self.names[start_index:end_index]

        array_features = []
        array_labels = []

        for i in range(len(batch_names)):
            name = batch_names[i]

            
            feature = cv2.cvtColor(
                cv2.imread(os.path.join(os.path.expanduser(self.folder_features), f"{name}.JPG"),
                           cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            
            label = io.loadmat(os.path.join(os.path.expanduser(self.folder_labels), f"{name}.mat"))["heatmap"]
            label = np.array(label).astype(np.float32) / 0.042

          
            if random.random() < self.flip_probability:
                feature = np.fliplr(feature)
                label = np.fliplr(label)

            array_features.append(feature)
            array_labels.append(label)

        array_features = np.array(array_features)
        array_labels = np.array(array_labels)

        return array_features, array_labels


class ShuffleDataCallback(Callback):
    def __init__(self, data_generator):
        super(ShuffleDataCallback, self).__init__()
        self.data_generator = data_generator

    def on_epoch_end(self, epoch, logs=None):
        
        random.shuffle(self.data_generator.names)




train_generator = CustomDataGenerator(train_folder_features, train_folder_labels, names_train, batch_size=1)
val_generator = CustomDataGenerator(val_folder_features, val_folder_labels, names_val, batch_size=1)


shuffle_data_callback = ShuffleDataCallback(train_generator)
shuffle_data_callback_val = ShuffleDataCallback(val_generator)

callbacks_list = [early_stopping, history_logger, shuffle_data_callback,shuffle_data_callback_val]



pretrained_model.fit(train_generator, validation_data=val_generator,epochs=EPOCHS, verbose=1,callbacks = callbacks_list)

pretrained_model.save(os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/PSP_DC_Soft_CSRNETP"))

print("Fitted Model")