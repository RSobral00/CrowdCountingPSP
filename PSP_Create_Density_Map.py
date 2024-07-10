# Rúben Sobral DensityMap Generator - PSPDataset

#####################################################################
#                                                                   #
#           CITAR CÓDIGO parte do density_map generation        #
#               https://arxiv.org/pdf/1912.01811.pdf                #
#                                                                   #
#####################################################################


import numpy as np
import scipy


def density_map_generator(image):
    

    def gaussian_filter_density(gt):
    
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        
        if gt_count == 0:
            return density

        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

       
        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = 1.

            sigma = 4

            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode="constant")
        
        return density

    d_map = gaussian_filter_density(image)
    
    return d_map

