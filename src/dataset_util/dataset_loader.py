import cv2
import math
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from jpeg import *
from dataset_util.paths import *


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

def load_minibatches(X_DATASET, Y_DATASET, mini_batch_size = 64):
    m = X_DATASET.shape[0]                 
    mini_batches = []

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X_DATASET[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = Y_DATASET[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X_DATASET[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = Y_DATASET[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def load_yuv_image(filePath):
    image_data = cv2.imread(filePath)
    image_data = cv2.resize(image_data, (IMG_DEFAULT_SIZE, IMG_DEFAULT_SIZE)) 
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2YUV)
    return image_data

def split_dataset(data_list, thr_1, thr_2):
    list_size = len(data_list)
    return data_list[: int(list_size*thr_1)], data_list[int(list_size*thr_1):int(list_size*thr_2)], data_list[int(list_size*thr_2):]


def dct_prob_map(dct):
    prob_map = np.zeros((1,IMG_DEFAULT_SIZE,IMG_DEFAULT_SIZE,1))
    for i in range(IMG_DEFAULT_SIZE):
        for j in range(IMG_DEFAULT_SIZE):
            if dct[i,j] >= 1 or dct[i,j] <= -1:
                prob_map[0,i,j,0] = 1

    return prob_map

def gen_x_and_y(SET):

    set_lenght = len(SET)

    qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
    qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

    x_set = np.zeros((set_lenght,IMG_DEFAULT_SIZE,IMG_DEFAULT_SIZE,6))
    y_set = np.zeros((set_lenght,IMG_DEFAULT_SIZE,IMG_DEFAULT_SIZE,6))

    for i, filePath in enumerate(SET):
        dct_100, dct_10 = load_dcts(filePath, ".png")
        
        x_set[i,:,:,:3] =  dct_10
        x_set[i,:,:,3:] =  decode_image(dct_10, qtable_luma_10, qtable_chroma_10, in_bgr=False)

        y_set[i,:,:,:3] = dct_100
        y_set[i,:,:,3:] = decode_image(dct_100, qtable_luma_100, qtable_chroma_100, in_bgr=False)

    
    return x_set, y_set

def insert_augmentation(data_list):
    new_data_list = []
    for data in data_list:
        new_data_list.append(data)
        file_x = data.split('.png')[0]+"_flip_x.png"
        new_data_list.append(file_x)
        file_y = data.split('.png')[0]+"_flip_y.png"
        new_data_list.append(file_y)
        file_xy = data.split('.png')[0]+"_flip_xy.png"
        new_data_list.append(file_xy)
    
    return new_data_list

def load_dataset(datasetDir):

    fileList = load_all_images_in_dir(datasetDir)

    #fileDic = split_by_category(fileList, ["1","2","3","4","5","6","7","8","9","10"], -2)
    fileDic = split_by_category(fileList, ["obama"], -2)
    TRAIN = []
    VALID = []
    TEST = []
    for category, fileList_byCat in fileDic.items():
        print("Class:", category, "Length:", len(fileList_byCat))
        train, valid, test = split_dataset(fileList_byCat,0.8,0.9)
        TRAIN += train
        VALID += valid
        TEST += test 
    
    TRAIN = insert_augmentation(TRAIN)
    TRAIN = np.asarray(TRAIN, dtype=np.unicode_)
    VALID = np.asarray(VALID, dtype=np.unicode_)
    TEST = np.asarray(TEST, dtype=np.unicode_)
    np.random.seed(0)
    permutation = list(np.random.permutation(len(TRAIN)))
    TRAIN = TRAIN[permutation]

    print("Sets size:\n", "train:", len(TRAIN), "valid:", len(VALID), "test:", len(TEST))

    TRAIN_X, TRAIN_Y = gen_x_and_y(TRAIN)
    #TRAIN.clear()

    VALID_X, VALID_Y = gen_x_and_y(VALID)
    #VALID.clear()

    TEST_X, TEST_Y = gen_x_and_y(TEST)
    #TEST.clear()

    print("Sets shape:\n", "train x:", TRAIN_X.shape, "y:", TRAIN_Y.shape,"\nvalid x:", VALID_X.shape, "y:", VALID_Y.shape, "\ntest x:", TEST_X.shape, "y:", TEST_Y.shape)

    return TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, TEST_X, TEST_Y

#load_dataset("../stl10/")

'''
dct_100, dct_10 = load_dcts("../stl10//2/0.bmp", ".bmp")

#view - debug
qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

dec_img_100 = decode_image(dct_100, qtable_luma_100, qtable_chroma_100)
dec_img_10 = decode_image(dct_10, qtable_luma_10, qtable_chroma_10)



stack = np.hstack([dec_img_100, dec_img_10])

cv2.imwrite('debug.jpg',stack)
'''