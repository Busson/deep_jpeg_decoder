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
ssim_10 = 0
first_frame = True
def evaluate_model(batch_10, batch_100, predict, write_out=False):
    global best_ssim
    batch_lenght = batch_10.shape[0]

    img100_patches = recover_patches(batch_100,100)
    img100 = merge_patches(img100_patches) 

    if first_frame:
        first_frame = False
        
        img10_patches = recover_patches(batch_10,10)
        img10 = merge_patches(img10_patches)

        ssim_10 = calc_ssim(img100, img10)

    imgP_patches = recover_predict(predict, quality1=100, quality2=10)
    imgP = merge_patches(imgP_patches)
    ssim_p = calc_ssim(img100, imgP) 

    if ssim_p > best_ssim:
        best_ssim = ssim_p

    print("SSIM 10:", ssim_10, "P:", ssim_p,"B:", best_ssim)

    return ssim_p