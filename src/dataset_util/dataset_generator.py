import cv2
from paths import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from jpeg import *


def create_dataset_dcts(file_name):

        ori_img = cv2.imread(file_name)
        ori_img = cv2.resize(ori_img, (IMG_DEFAULT_SIZE, IMG_DEFAULT_SIZE)) 

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

        file_100_basename = file_name.split('.png')[0]+"_dct_100.txt"
        file_10_basename = file_name.split('.png')[0]+"_dct_10.txt"

        if os.path.isfile(file_100_basename):
                os.remove(file_100_basename)

        if os.path.isfile(file_10_basename):
                os.remove(file_10_basename)

        file_100 = open(file_100_basename, "w")
        file_100.write(str_img_100_0[:-1]+"\n")
        file_100.write(str_img_100_1[:-1]+"\n")
        file_100.write(str_img_100_2[:-1]+"\n")
        file_100.close()

        file_10 = open(file_10_basename, "w")
        file_10.write(str_img_10_0[:-1]+"\n")
        file_10.write(str_img_10_1[:-1]+"\n")
        file_10.write(str_img_10_2[:-1]+"\n")
        file_10.close()


def augment_dataset(listOfFiles):
        new_file_list = []
        for data in listOfFiles:
                new_file_list.append(data)

                ori_img = cv2.imread(data)
                ori_img = cv2.resize(ori_img, (IMG_DEFAULT_SIZE, IMG_DEFAULT_SIZE)) 

                file_x = data.split('.png')[0]+"_flip_x.png"
                new_file_list.append(file_x)
                file_y = data.split('.png')[0]+"_flip_y.png"
                new_file_list.append(file_y)
                file_xy = data.split('.png')[0]+"_flip_xy.png"
                new_file_list.append(file_xy)

                if os.path.isfile(file_x):
                        os.remove(file_x)

                if os.path.isfile(file_y):
                        os.remove(file_y)

                if os.path.isfile(file_xy):
                        os.remove(file_xy)

                
                flip_x_img = cv2.flip(ori_img, 0)
                flip_y_img = cv2.flip(ori_img, 1)
                flip_xy_img = cv2.flip(ori_img, -1)

                cv2.imwrite(file_x,flip_x_img)
                cv2.imwrite(file_y,flip_y_img)
                cv2.imwrite(file_xy,flip_xy_img)

        return new_file_list


listOfFiles = load_all_images_in_dir("../stl10/")

listOfFiles = augment_dataset(listOfFiles)

for data in listOfFiles:
        print(data)
        create_dataset_dcts(data)
