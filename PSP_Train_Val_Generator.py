#PSP Train Val Generator 

import os 
import json 
import cv2
import numpy as np 
import scipy.io as io 
import PSP_Create_Density_Map

gt_annotation_path = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/GT_annotation")

images_path = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/train_images")
#images_path = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/val_images")
labels_path = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/train_labels")
#labels_path = os.path.expanduser("~/Desktop/Tese/PSP_Dataset/val_labels")


scale_factor_y = 1080 / 2280
scale_factor_x = 1920 / 4056 

for gt_file in os.listdir(gt_annotation_path):

    gt_full_path = os.path.join(gt_annotation_path,gt_file)
    with open(gt_full_path, "r") as file:

        content = file.read()

        # Parse the JSON content
        data = [json.loads(line) for line in content.split("\n") if line.strip()]


        for entry in data:

            external_id = entry["data_row"]["external_id"]

           
      

            image_path = os.path.join(images_path, external_id)

            if os.path.exists(image_path):
                
                print(f"Image {external_id}: Generating.")


                coordinates = []

                for annotation in entry["projects"]["clrzfkqls00yi07wp3ndhb7nx"]["labels"][0]["annotations"]["objects"]:
                    if annotation["name"] == "People":
                        point = annotation["point"]
                        coordinates.append((point["y"],point["x"]))


            
                

                if "C1_"  in external_id or "C2_"  in external_id or "C3_" in external_id:

                    new_coordinates = []

                    for coord in coordinates:
                        new_coord = (round(coord[0]), round(coord[1]))
                        new_coordinates.append(new_coord)


                else:

                    new_coordinates = []

                    for coord in coordinates:
                        new_coord = (round(coord[0] * scale_factor_y), round(coord[1] * scale_factor_x))
                        new_coordinates.append(new_coord)




            

                img_bin = np.zeros((1080,1920), dtype=np.float32)

                for y,x in new_coordinates:

                    img_bin[y-1,x-1]=1 
                    

                d_map = PSP_Create_Density_Map.density_map_generator(img_bin)

                label_filename = external_id.replace(".JPG", ".mat")
                label_path = os.path.join(labels_path, label_filename)
                io.savemat(label_path, {"heatmap": d_map})
                
                print(f"{external_id}: Saved. \n")


        else:

            print(f"Image {external_id}: Skipping...")
            continue
            
            


print("Done...")