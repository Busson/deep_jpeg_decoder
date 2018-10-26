import cv2
from jpeg import *


ori_dataset_folder="images/"


file_name="images/parrot.bmp" 

ori_img = cv2.imread(file_name)

qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

dct_img_100 = encode_image(ori_img, qtable_luma_100, qtable_chroma_100)
dct_img_10 = encode_image(ori_img, qtable_luma_10, qtable_chroma_10)

str_img_100_0 = ""
str_img_100_1 = ""
str_img_100_2 = ""

str_img_10_0 = ""
str_img_10_1 = ""
str_img_10_2 = ""

for i in range(0, ori_img.shape[0]):
    for j in range(0, ori_img.shape[1]):
        
        str_img_100_0 += str(dct_img_100[i,j,0])+";"
        str_img_100_1 += str(dct_img_100[i,j,1])+";"
        str_img_100_2 += str(dct_img_100[i,j,2])+";"
        
        #print(dct_img_10[i,j,0],";")
        str_img_10_0 += str(dct_img_10[i,j,0])+";"
        str_img_10_1 += str(dct_img_10[i,j,1])+";"
        str_img_10_2 += str(dct_img_10[i,j,2])+";"


file_100 = open("images/parrot_dct_100.txt", "w")
file_100.write(str_img_100_0[:-1]+"\n")
file_100.write(str_img_100_1[:-1]+"\n")
file_100.write(str_img_100_2[:-1]+"\n")
file_100.close()

file_10 = open("images/parrot_dct_10.txt", "w")
file_10.write(str_img_10_0[:-1]+"\n")
file_10.write(str_img_10_1[:-1]+"\n")
file_10.write(str_img_10_2[:-1]+"\n")
file_10.close()

