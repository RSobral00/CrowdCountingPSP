#Rúben Sobral 93273 UA - DFIS - Tese
#Train Data Generator (HEATMAPS .mat)

import os 
import scipy.io as io
import numpy as np 
import pandas as pd 
import DC_Create_DensityMap_Gaussian

height, width = (1080,1920)

#Write train information 
train_info = os.path.expanduser("~/Desktop/Tese/DroneCrowd/train_data/train_info.txt")



with open(train_info, "a") as file:

    #Open trainlist.txt
    with open(os.path.expanduser("~/Desktop/Tese/DroneCrowd/trainlist.txt"), "r") as train_list:    
        lines = train_list.readlines() 


    
        for seq in lines:  # Sequence Number

            seq = seq.strip("\n")

           

        
            print("---------seq--------:",seq)

           #Annotation folder + file .mat containing the coordinates
            mat_file_path = os.path.expanduser("~/Desktop/Tese/DroneCrowd/annotations/" + str(seq) + ".mat")
            
            mat_contents = io.loadmat(mat_file_path)

           
            array_seq = mat_contents["anno"] 

            col_names = ["Frame","Id" ,"Xmin", "Ymin", "Xmax", "Ymax"]

            data_frame = pd.DataFrame(array_seq, columns=col_names) 

            data_frame =  data_frame.reindex(columns=["Id","Frame","Xmin", "Ymin", "Xmax", "Ymax"])

            data_frame["Center"] = data_frame.apply(lambda row: (round(0.5 * (row["Ymin"] + row["Ymax"])), round(0.5 * (row["Xmin"] + row["Xmax"]))), axis=1)

            max_frame = data_frame["Frame"].max()


            f = 0 

            for frame in range(max_frame+1):

                image_frame = data_frame[data_frame["Frame"] == f]    

                coord_list = list(image_frame["Center"])
        

                density_ma_bin = np.zeros((height, width), dtype=np.float32)

                

                for x,y in coord_list:

                    if x > 1080:
                        x = 1080 

                    if y > 1920:
                        y = 1920 

                    density_ma_bin[x-1,y-1]=1 
                    
                #Function to generate the density map
                heatmap = DC_Create_DensityMap_Gaussian.heatmap_generator(density_ma_bin)

                count = np.sum(density_ma_bin)
                sum_heatmap = np.sum(heatmap)


                
                f += 1  
                num_seq = seq[2:]
                num_f_save = f
                num_f_save = "{:03d}".format(num_f_save)
                num_seq_frame = str(num_seq) + str(num_f_save)
                mat_file_path = os.path.expanduser("~/Desktop/Tese/DroneCrowd/train_data/heatmaps_train/" + str(num_seq_frame) +".mat")

                # Save the NumPy ndarray in a .mat file
                io.savemat(mat_file_path, {"heatmap": heatmap})

                #Saves information about the total pedestrian count for each train image.
                file.write("\n" + str(num_seq_frame) + "," + str(count) + "," + str(sum_heatmap))

                if f==1 or  f==150 or f==300:   
                    print("f:",f)
            

file.close()