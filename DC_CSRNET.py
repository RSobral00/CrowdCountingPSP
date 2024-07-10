#Rúben Sobral 93273 UA - DFIS - Tese
# CSRNET DroneCrowd implementation - https://github.com/CS3244-AY2021-SEM-1/csrnet-tensorflow/blob/master/CSRNet_Keras.ipynb


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

########################### RUN WITH GPU ############################################
#Check Available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Name GPUs Available: ", tf.config.experimental.list_physical_devices("GPU"))

# Enable Memory Growth On Both GPUs
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

#######################################################################################




# Set random seed for Python environment
seed = 42
random.seed(seed)
np.random.seed(seed)

# Set random seed for TensorFlow
tf.random.set_seed(seed)

# tf.keras.backend.set_floatx("float32")



#Define file paths
train_folder_features = "~/Desktop/Tese/DroneCrowd/train_data/images_train"
train_folder_labels = "~/Desktop/Tese/DroneCrowd/train_data/heatmaps_train"
val_folder_features = "~/Desktop/Tese/DroneCrowd/val_data/images_val"         
val_folder_labels = "~/Desktop/Tese/DroneCrowd/val_data/heatmaps_val"


# List file names and shuffle them
names_train = os.listdir(os.path.expanduser(train_folder_labels))
names_val = os.listdir(os.path.expanduser(val_folder_labels))

names_train = [os.path.splitext(file_name)[0] for file_name in names_train]
names_val = [os.path.splitext(file_name)[0] for file_name in names_val]




random.shuffle(names_train )
random.shuffle(names_val)

print("---names_train generation completed---")
print("---names_val generation completed---")


def init_weights_vgg(model):
    #vgg =  VGG16(weights='imagenet', include_top=False)
    
    json_path = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/VGG_16.json") #VGG16 

    with open(json_path, 'r') as f:
        loaded_model_json = f.read()

    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights(os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/VGG_16.h5")) #VGG16 Pre-trained parameters
    
    vgg = loaded_model
    
    vgg_weights=[]                         
    for layer in vgg.layers:
        if('conv' in layer.name):
            vgg_weights.append(layer.get_weights())
    
    
    offset=0
    i=0
    while(i<10):
        if('conv' in model.layers[i+offset].name):
            model.layers[i+offset].set_weights(vgg_weights[i])
            i=i+1
            #print('h')
            
        else:
            offset=offset+1

    return (model)
    


# Neural network model : VGG + Conv
def CrowdNet():  
           
            rows = 1080
            cols = 1920
            
          
            
            batch_norm = 1
            kernel = (3, 3)
            init = RandomNormal(stddev=0.01)
            model = Sequential() 
            
            #custom VGG:
            
            if(batch_norm):
                model.add(Conv2D(64, kernel_size = kernel, input_shape = (rows,cols,3),activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(strides=2))
                model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(strides=2))
                model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(strides=2))            
                model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
                model.add(BatchNormalization())
                
            else:
                model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same',input_shape = (rows, cols, 3), kernel_initializer = init))
                model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
                model.add(MaxPooling2D(strides=2))
                model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
                model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
                model.add(MaxPooling2D(strides=2))
                model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
                model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
                model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
                model.add(MaxPooling2D(strides=2))            
                model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
                model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
                model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
                
                

                
            #Conv2D
            model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
            model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
            model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
            model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
            model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
            model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
            model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate = 1, kernel_initializer = init, padding = 'same'))
            model.add(UpSampling2D((8,8),interpolation="bilinear"))
            
            
            
            model = init_weights_vgg(model)
            
            return model



model = CrowdNet()
model.summary()





checkpoint_filepath = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/CSRNET_Model")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
    save_best_only=True
)




opt = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.95)

def euclidean_distance_loss(y_true, y_pred):

    return K.square(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


EPOCHS = 60

model.compile(optimizer=opt,   
              loss=euclidean_distance_loss,  
              metrics=["mse", "mae"])  


filename = os.path.expanduser("~/Desktop/Tese/CrowdCountingPSP/Results/CSRNET_Model_history.csv")
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
                cv2.imread(os.path.join(os.path.expanduser(self.folder_features), f"img{name}.jpg"),
                           cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            
            label = io.loadmat(os.path.join(os.path.expanduser(self.folder_labels), f"{name}.mat"))["heatmap"]
            label = np.array(label).astype(np.float32) / 0.042 #On inference images it is needed to multiply the predicted DM by 0.042 to obtain the count
                                                            ##Normalized with the maximum value of all the training density maps
          
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

callbacks_list = [model_checkpoint, history_logger, shuffle_data_callback,shuffle_data_callback_val]

history = model.fit(train_generator, validation_data=val_generator,epochs=EPOCHS, verbose=1,callbacks = callbacks_list)

print("Fitted Model")

