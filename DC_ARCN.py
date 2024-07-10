# Rúben Sobral ARCN implementation 

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
#Check Available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Name GPUs Available: ", tf.config.experimental.list_physical_devices("GPU"))
print("GPU_INFO",tf.config.list_physical_devices('GPU'))

#Check Available GPUs

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



#################################### Functions ####################################################

########## Bottlenecks function #########

def relu6(x):
    return K.relu(x, max_value=6)

def bottleneck_block(inputs, c, s, t):
    channels = int(inputs.shape[-1])

    x = Conv2D(t * channels, (1, 1), (1, 1), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = DepthwiseConv2D((3, 3), strides=s, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = Conv2D(c, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    if s == 1 and channels == c:
        x = Add()([x, inputs])

    return x


################ CBAM MODULE #######################

def channel_gate(x, reduction_ratio=16):
    gate_channels = x.shape[-1]
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)

    avg_mlp = tf.keras.Sequential([
        Dense(gate_channels // reduction_ratio, activation='relu'),
        Dense(gate_channels)
    ])
    max_mlp = tf.keras.Sequential([
        Dense(gate_channels // reduction_ratio, activation='relu'),
        Dense(gate_channels)
    ])

    channel_att_raw = avg_mlp(avg_pool) + max_mlp(max_pool)
    scale = tf.math.sigmoid(channel_att_raw)

    return x * scale

def spatial_gate(x):
    kernel_size = 7
    compress = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    spatial_att = tf.keras.Sequential([
        Conv2D(1, kernel_size=kernel_size, strides=1, padding='same'),
        Activation('sigmoid')
    ])

    scale = spatial_att(compress)
    return x * scale

def cbam_module(x, reduction_ratio=16):
    channel_att = channel_gate(x, reduction_ratio)
    spatial_att = spatial_gate(channel_att)
    return spatial_att

############# CRP Block #############

def crp_layer(filters, inputs, num_reps=4):

    block_outputs = []

    x = inputs  
    
    for _ in range(num_reps):
        pool_block = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="same")(x)
        conv_block = Conv2D(filters, (1, 1), padding="same", activation=None)(pool_block)
        block_outputs.append(conv_block)
        x = conv_block

 
    x = Conv2D(filters, (1, 1), activation=None)(x)


    output = x + tf.reduce_sum(block_outputs, axis=0)
    return output



############## Fusion Block ################

def fusion_block(high_resolution, low_resolution):


    high_resolution = Conv2D(high_resolution.shape[-1], kernel_size=(1, 1))(high_resolution)
    

    low_resolution = Conv2D(high_resolution.shape[-1], kernel_size=(1, 1), activation=None)(low_resolution)

    target_shape = K.int_shape(high_resolution)[1:3]
    
    low_resolution = tf.image.resize(low_resolution, size=(target_shape[0], target_shape[1]), method=tf.image.ResizeMethod.BILINEAR)

    height_cropping = max(0, low_resolution.shape[1] - target_shape[0])
    width_cropping = max(0, low_resolution.shape[2] - target_shape[1])
    low_resolution = Cropping2D(cropping=((0, height_cropping), (0, width_cropping)))(low_resolution)

    fused_feature = Add()([high_resolution, low_resolution])
    fused_feature = Activation("relu")(fused_feature)
    
    return fused_feature






################### Prediction Layer ####################################################


def density_prediction_layer(inputs,target_shape=(1080, 1920)):
    
    
    density_map = Conv2D(1, kernel_size=(1, 1), activation="linear", padding="same")(inputs)

    target_height, target_width = target_shape
    

    predicted_density_map = tf.image.resize(density_map, size=(target_height, target_width), method=tf.image.ResizeMethod.BILINEAR)
    

    return predicted_density_map




#########################################                   ###############################################



# CNN Structure 

input_layer = Input(shape=(1080,1920,3))

# conv1 = Conv2D(32, (3, 3),(2,2),padding="same",activation="relu",kernel_initializer="glorot_uniform")(input_layer)
conv1 = Conv2D(32, (3, 3),(2,2),padding="same",activation="relu")(input_layer)

# batch_norm1 = BatchNormalization()(conv1)
# relu1 = Activation("relu")(batch_norm1)

pool1 = MaxPooling2D((3,3),(2,2),padding="same")(conv1)


#Bottleneck 1
expansion1 = 1  
num_repeats1 = 1
stride1 = 1
filters1 = 32


x = pool1
for _ in range(num_repeats1):
    x = bottleneck_block(x, filters1,stride1,expansion1)
   
bottleneck1 = x 


#CBAM1
cbam1 = cbam_module(bottleneck1)


#Bottleneck 2
expansion2 = 6  
num_repeats2 = 2
stride2 = 2
filters2 = 64


x = cbam1
for _ in range(num_repeats2):
    x = bottleneck_block(x, filters2,stride2,expansion2)
    

bottleneck2 = x

#CBAM2
cbam2 = cbam_module(bottleneck2)


#Bottleneck 3
expansion3 = 6  
num_repeats3 = 3
stride3 = 2
filters3 = 128


x = cbam2
for _ in range(num_repeats3):
    x = bottleneck_block(x, filters3,stride3,expansion3)
    

bottleneck3 = x

#CBAM3
cbam3 = cbam_module(bottleneck3)
cbam3 = Dropout(0.5)(cbam3)

#Bottleneck 4

expansion4 = 6  
num_repeats4 = 4
stride4 = 2
filters4 = 256
#filters4 = 512

x = cbam3
for _ in range(num_repeats4):
    x= bottleneck_block(x, filters4,stride4,expansion4)
    

bottleneck4 = x



#CBAM4 
cbam4 = cbam_module(bottleneck4)
cbam4 = Dropout(0.5)(cbam4)


# Decoder part LW Refine Net

#CRP1

crp1 = crp_layer(64,cbam4)



#Fusion1


fusion1 = fusion_block(cbam3, crp1)


#CRP2 
crp2 = crp_layer(32,fusion1)



#Fusion2


fusion2 = fusion_block(cbam2, crp2)

#CRP3
crp3 = crp_layer(32,fusion2)



#Fusion3
fusion3 = fusion_block(cbam1, crp3)


#CRP4
crp4 = crp_layer(32,fusion3)
crp4 = Dropout(0.5)(crp4)

predicted_density_map = density_prediction_layer(crp4)

# Create a model
model = tf.keras.Model(input_layer, outputs=predicted_density_map)
model.summary()





# checkpoint_filepath = os.path.expanduser("~/Desktop/Tese/CrowdCountingUAV/Results/ARCN_Model30_RF")
# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor="val_loss",
#     mode="min",
#     save_best_only=True
# )



opt =tfa.optimizers.AdamW(learning_rate=1.25e-5,weight_decay=1.25e-5)        



def euclidean_distance_loss(y_true, y_pred):

    return K.square(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


EPOCHS = 60

model.compile(optimizer=opt,   
              loss=euclidean_distance_loss,  
              metrics=["mse", "mae"])  

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


filename = os.path.expanduser("~/Desktop/Tese/CrowdCountingUAV/Results/ARCN_Model_history.csv")
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)


# Data Generator - Loads Dataset in batches
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

            # Load and preprocess feature (image)
            feature = cv2.cvtColor(
                cv2.imread(os.path.join(os.path.expanduser(self.folder_features), f"img{name}.jpg"),
                           cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            # Load and preprocess label
            label = io.loadmat(os.path.join(os.path.expanduser(self.folder_labels), f"{name}.mat"))["heatmap"]
            
            label = np.array(label).astype(np.float32) / 0.042
            

            # # Randomly flip the image and label
            # if random.random() < self.flip_probability:
            #     feature = np.fliplr(feature)
            #     label = np.fliplr(label)

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




train_generator = CustomDataGenerator(train_folder_features, train_folder_labels, names_train, batch_size=4)
val_generator = CustomDataGenerator(val_folder_features, val_folder_labels, names_val, batch_size=4)


shuffle_data_callback = ShuffleDataCallback(train_generator)
shuffle_data_callback_val = ShuffleDataCallback(val_generator)

# callbacks_list = [model_checkpoint, history_logger, shuffle_data_callback,shuffle_data_callback_val]
callbacks_list = [early_stopping, history_logger, shuffle_data_callback,shuffle_data_callback_val]
history = model.fit(train_generator, validation_data=val_generator,epochs=EPOCHS, verbose=1,callbacks = callbacks_list)




# Save the trained model
model.save(os.path.expanduser("~/Desktop/Tese/CrowdCountingUAV/Results/ARCN_Model"))


print("Fitted Model")
