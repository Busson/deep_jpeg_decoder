import cv2
import math
import numpy as np
from jpeg import *


def load_dcts(file_name, etx_sep):
    base_name = file_name.split(etx_sep)[0]

    file_100 = open(base_name+"_dct_100.txt", "r")
    data_100 = file_100.readlines()

    file_10 = open(base_name+"_dct_10.txt", "r")
    data_10 = file_10.readlines()

    num_100_0 = data_100[0].split(";")
    num_100_1 = data_100[1].split(";")
    num_100_2 = data_100[2].split(";")

    num_10_0 = data_10[0].split(";")
    num_10_1 = data_10[1].split(";")
    num_10_2 = data_10[2].split(";")

    map_size = int(math.sqrt(len(num_100_0)))

    dct_100 = np.zeros((map_size,map_size,3))
    dct_10 = np.zeros((map_size,map_size,3))

    for i in range(0, map_size):
        for j in range(0, map_size):
            dct_100[i,j,0] = float(num_100_0[(i*map_size)+j])
            dct_100[i,j,1] = float(num_100_1[(i*map_size)+j])
            dct_100[i,j,2] = float(num_100_2[(i*map_size)+j])

            dct_10[i,j,0] = float(num_10_0[(i*map_size)+j])
            dct_10[i,j,1] = float(num_10_1[(i*map_size)+j])
            dct_10[i,j,2] = float(num_10_2[(i*map_size)+j])



    return dct_100, dct_10

'''
dct_100, dct_10 = load_dcts("images/lena.bmp", ".bmp")

#view - debug
qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

dec_img_100 = decode_image(dct_100, qtable_luma_100, qtable_chroma_100)
dec_img_10 = decode_image(dct_10, qtable_luma_10, qtable_chroma_10)



stack = np.hstack([dec_img_100, dec_img_10])

cv2.imshow('image',stack)
cv2.waitKey(0)
'''