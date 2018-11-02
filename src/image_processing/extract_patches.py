import cv2
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from jpeg import *


PATCH_SIZE = 96
FRAME_ASP_WIDTH = 16
FRAME_ASP_HEIGHT = 9

def extract_patches(filePath):
    video_frame = cv2.imread(filePath)
    
    if video_frame.shape[0]%PATCH_SIZE != 0 or video_frame.shape[1]%PATCH_SIZE:
        print("Incompatible frame dimension")
        return

    frame_width = int(video_frame.shape[1]/PATCH_SIZE)
    frame_height = int(video_frame.shape[0]/PATCH_SIZE)
    batch_size = frame_width*frame_height
    batch = np.zeros((batch_size,PATCH_SIZE,PATCH_SIZE,3), dtype=np.uint8)
    i=0
    for x in range(0,video_frame.shape[0], PATCH_SIZE):
        for y in range(0,video_frame.shape[1], PATCH_SIZE):
            batch[i,:,:,:] = video_frame[x:x+PATCH_SIZE, y:y+PATCH_SIZE, :]
            i += 1

    return batch

def merge_patches(batch):

    batch_size = batch.shape[0]

    frame_width =  FRAME_ASP_WIDTH*PATCH_SIZE
    frame_height = FRAME_ASP_HEIGHT*PATCH_SIZE
    i=0
    frame = np.zeros((frame_height,frame_width,3), dtype=np.uint8)

    for x in range(0, frame_height, PATCH_SIZE):
        for y in range(0, frame_width, PATCH_SIZE):
            frame[x:x+PATCH_SIZE, y:y+PATCH_SIZE, :] = batch[i]
            i += 1

    return frame


def encode_decode_patches(batch, quality=10):
    batch = batch.copy()
    qtable_luma, qtable_chroma = generate_qtables(quality_factor=quality)
    for i in range(batch.shape[0]):
        dct = encode_image(batch[i,:,:,:], qtable_luma, qtable_chroma)
        batch[i,:,:,:] = decode_image(dct, qtable_luma, qtable_chroma)
        
    return batch

def get_patches_dcts(batch, quality1=100, quality2=10):

    batch_1 = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))
    batch_2 = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))

    qtable_luma1, qtable_chroma1 = generate_qtables(quality_factor=quality1)
    qtable_luma2, qtable_chroma2 = generate_qtables(quality_factor=quality2)

    for i in range(batch.shape[0]):
        batch_1[i,:,:,:] = encode_image(batch[i,:,:,:], qtable_luma1, qtable_chroma1)
        batch_2[i,:,:,:] = encode_image(batch[i,:,:,:], qtable_luma2, qtable_chroma2)
    
    return batch_1, batch_2

def recover_patches(dcts, quality): 
    qtable_luma, qtable_chroma = generate_qtables(quality_factor=quality)
    batch = np.zeros((dcts.shape[0], dcts.shape[1], dcts.shape[2], dcts.shape[3]))
    for i in range(dcts.shape[0]):
        batch[i,:,:,:] = decode_image(dcts[i,:,:,:], qtable_luma, qtable_chroma)
        
    return batch


'''
batch = extract_patches("images/frame1.jpeg")
#batch = encode_decode_patches(batch)
dct100, dct10 = get_patches_dcts(batch)
print("dcts:",dct100.shape, dct10.shape)
img100 = recover_patches(dct100,100)
img10 = recover_patches(dct10,10)

cv2.imshow('image100', merge_patches(img100))
cv2.imshow('image10', merge_patches(img10))
cv2.waitKey(0)




for i in range(batch.shape[0]):
    cv2.imshow('image', batch[i,:,:,:])
    cv2.waitKey(0)

'''