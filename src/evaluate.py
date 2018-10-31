import numpy as np 
from skimage.measure import compare_ssim, compare_psnr, compare_nrmse
import cv2
from jpeg import *


def calc_nrmse(img_ref, img_test):
    return compare_nrmse(img_ref, img_test)

def calc_ssim(img_ref, img_test):
    return compare_ssim(img_ref, img_test, multichannel=True)

def calc_psnr(img_ref, img_test):
    return compare_psnr(img_ref, img_test, data_range=255)


def evaluate_model(batch_x, batch_y, predict, write_out=False):

    batch_lenght = batch_x.shape[0]

    #print(batch_lenght)

    qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
    qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

    imgs_q_100 = np.zeros((batch_lenght,96,96,3))
    imgs_pred = np.zeros((batch_lenght,96,96,3))

    mean_ssim_p = 0
    mean_ssim_10 = 0

    stackList = [] 

    for i in range(batch_lenght):

        dct_10 = batch_x[i,:,:,:]
        dct_100 = batch_y[i,:,:,:]
        dct_100[:,:,:] = dct_100[:,:,:] + dct_10[:,:,:]

        dct_pred = np.zeros((96,96,3))
        dct_pred[:,:,0] = predict[i,:,:,0] + dct_10[:,:,0]
        dct_pred[:,:,1] = dct_10[:,:,1]
        dct_pred[:,:,2] = dct_10[:,:,2]

        dec_q_100 = decode_image(dct_100, qtable_luma_100, qtable_chroma_100)
        dec_q_10 = decode_image(dct_10, qtable_luma_10, qtable_chroma_10)
        dec_pred = decode_image(dct_pred, qtable_luma_100, qtable_chroma_10)

        mean_ssim_p += calc_ssim(dec_q_100, dec_pred)
        mean_ssim_10 += calc_ssim(dec_q_100, dec_q_10)

        if write_out == True:
            stackList.append(np.hstack([dec_q_100, dec_q_10, dec_pred]))

    
    mean_ssim_p = float(mean_ssim_p/batch_lenght)
    mean_ssim_10 = float(mean_ssim_10/batch_lenght)

    print("MEAN SSIM P:", mean_ssim_p, "SSIM 10:", mean_ssim_10)

    
    if write_out == True:
        stack = np.vstack(stackList)
        cv2.imwrite("evaluate.jpg", stack)