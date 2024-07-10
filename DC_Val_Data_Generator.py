#Rúben Sobral 93273 UA - DFIS - Tese
#Val Data Generator (HEATMAPS .mat) (Copy Files from Test data) 
#All the val_heatmaps are contained in the Test data already 

import os 
import shutil 

val_folder_images = os.path.expanduser("~/Desktop/Tese/DroneCrowd/val_data/images_val") #validation images

val_folder_heatmaps = os.path.expanduser("~/Desktop/Tese/DroneCrowd/val_data/heatmaps_val") #empty folder of val_heatmaps

test_folder_heatmaps = os.path.expanduser("~/Desktop/Tese/DroneCrowd/test_data/heatmaps_test")  # copying from test_heatmaps 

# List file names and shuffle them
names_val_images= os.listdir(val_folder_images)

num_val_img = [os.path.splitext(file_name)[0] for file_name in names_val_images]
num_val_img = [file_name[3:] for file_name in num_val_img]


for filename in num_val_img:
    source_path = os.path.join(test_folder_heatmaps, filename + ".mat")
    destination_path = os.path.join(val_folder_heatmaps, filename + ".mat")

    # Check if the file exists before copying
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"File {filename}.mat copied successfully.")
    else:
        print(f"File {filename}.mat does not exist in the source folder.")
