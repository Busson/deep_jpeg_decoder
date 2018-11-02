import numpy as np 
from skimage.measure import compare_ssim, compare_psnr, compare_nrmse
import cv2
from jpeg import *
from image_processing.extract_patches import *


def calc_nrmse(img_ref, img_test):
    return compare_nrmse(img_ref, img_test)

def calc_ssim(img_ref, img_test):
    return compare_ssim(img_ref, img_test, multichannel=True)

def calc_psnr(img_ref, img_test):
    return compare_psnr(img_ref, img_test, data_range=255)


best_ssim = 0

def evaluate_model(batch_x, batch_y, predict, write_out=False):
    global best_ssim
    batch_lenght = batch_x.shape[0]

    #print(batch_lenght)

    qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
    qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

    imgs_q_100 = np.zeros((batch_lenght,IMG_DEFAULT_SIZE,IMG_DEFAULT_SIZE,3))
    imgs_pred = np.zeros((batch_lenght,IMG_DEFAULT_SIZE,IMG_DEFAULT_SIZE,3))

    mean_ssim_p = 0
    mean_ssim_10 = 0

    mean_psnr_p = 0
    mean_psnr_10 = 0

    mean_nrmse_p = 0
    mean_nrmse_10 = 0

    stackList = [] 

    for i in range(batch_lenght):

        dct_10 = batch_x[i,:,:,:3]
        dct_100 = batch_y[i,:,:,:3]
    
        dct_pred = np.zeros((IMG_DEFAULT_SIZE,IMG_DEFAULT_SIZE,3))
        #dct_pred[:,:,0] = predict[i,:,:,0] + dct_10[:,:,0]
        dct_pred[:,:,0] = predict[i,:,:,0]
        dct_pred[:,:,1] = dct_10[:,:,1]
        dct_pred[:,:,2] = dct_10[:,:,2]

        #print(dct_pred[:4,:4,0])
        #print(" ")
        #print(dct_100[:4,:4,0])

        dec_q_100 = decode_image(dct_100, qtable_luma_100, qtable_chroma_100)
        dec_q_10 = decode_image(dct_10, qtable_luma_10, qtable_chroma_10)
        dec_pred = decode_image(dct_pred, qtable_luma_100, qtable_chroma_10)

        mean_ssim_p += calc_ssim(dec_q_100, dec_pred)
        mean_ssim_10 += calc_ssim(dec_q_100, dec_q_10)

        mean_psnr_p += calc_psnr(dec_q_100, dec_pred)
        mean_psnr_10 += calc_psnr(dec_q_100, dec_q_10)

        mean_nrmse_p += calc_nrmse(dec_q_100, dec_pred)
        mean_nrmse_10 += calc_nrmse(dec_q_100, dec_q_10)

        cv2.imwrite("test"+str(i)+".jpg", np.hstack([batch_x[i,:,:,3:4],batch_x[i,:,:,:1], batch_y[i,:,:,3:4], batch_y[i,:,:,:1]]))

        if write_out == True:
            stackList.append(np.hstack([dec_q_100, dec_q_10, dec_pred]))

    
    mean_ssim_p = float(mean_ssim_p/batch_lenght)
    mean_ssim_10 = float(mean_ssim_10/batch_lenght)

    if mean_ssim_p > best_ssim:
        best_ssim = mean_ssim_p

    mean_psnr_p = float(mean_psnr_p/batch_lenght)
    mean_psnr_10 = float(mean_psnr_10/batch_lenght)

    mean_nrmse_p = float(mean_nrmse_p/batch_lenght)
    mean_nrmse_10 = float(mean_nrmse_10/batch_lenght)
    
    print("SSIM P:", mean_ssim_p, "10:", mean_ssim_10, "BEST:", best_ssim)
    print("PSNR P:", mean_psnr_p, "10:", mean_psnr_10)
    print("NRMSE P:", mean_nrmse_p, "10:", mean_nrmse_10)
    
    if write_out == True:
        stack = np.vstack(stackList)
        cv2.imwrite("evaluate.jpg", stack)

    
    return mean_ssim_p