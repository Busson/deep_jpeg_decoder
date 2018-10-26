import cv2
from jpeg import *
from evaluate import *

#exemplo de uso
ori_img = cv2.imread("images/lena_96.bmp")

qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
qtable_luma_50, qtable_chroma_50 = generate_qtables(quality_factor=50)
qtable_luma_25, qtable_chroma_25 = generate_qtables(quality_factor=25)
qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)
qtable_luma_5, qtable_chroma_5 = generate_qtables(quality_factor=5)

dct_img_100 = encode_image(ori_img, qtable_luma_100, qtable_chroma_100)
dct_img_50 = encode_image(ori_img, qtable_luma_50, qtable_chroma_50)
dct_img_25 = encode_image(ori_img, qtable_luma_25, qtable_chroma_25)
dct_img_10 = encode_image(ori_img, qtable_luma_10, qtable_chroma_10)
dct_img_5 = encode_image(ori_img, qtable_luma_5, qtable_chroma_5)


dct_img_sp = np.zeros((dct_img_100.shape[0],dct_img_100.shape[1],3))
dct_img_sp[:,:,0] = dct_img_100[:,:,0]
dct_img_sp[:,:,1] = dct_img_10[:,:,1]
dct_img_sp[:,:,2] = dct_img_10[:,:,2]

dec_img_100 = decode_image(dct_img_100, qtable_luma_100, qtable_chroma_100)
dec_img_50 = decode_image(dct_img_50, qtable_luma_50, qtable_chroma_50)
dec_img_25 = decode_image(dct_img_25, qtable_luma_25, qtable_chroma_25)
dec_img_10 = decode_image(dct_img_10, qtable_luma_10, qtable_chroma_10)
dec_img_5 = decode_image(dct_img_5, qtable_luma_5, qtable_chroma_5)

dec_img_sp = decode_image(dct_img_sp, qtable_luma_100, qtable_chroma_10)


print("100 - NRMSE:", calc_nrmse(dec_img_100, dec_img_100), 
"SSIM:", calc_ssim(dec_img_100, dec_img_100),
"PSNR:", calc_psnr(dec_img_100, dec_img_100))

print("50 - NRMSE:", calc_nrmse(dec_img_100, dec_img_50), 
"SSIM:", calc_ssim(dec_img_100, dec_img_50),
"PSNR:", calc_psnr(dec_img_100, dec_img_50))

print("25 - NRMSE:", calc_nrmse(dec_img_100, dec_img_25), 
"SSIM:", calc_ssim(dec_img_100, dec_img_25),
"PSNR:", calc_psnr(dec_img_100, dec_img_25))

print("10 - NRMSE:", calc_nrmse(dec_img_100, dec_img_10), 
"SSIM:", calc_ssim(dec_img_100, dec_img_10),
"PSNR:", calc_psnr(dec_img_100, dec_img_10))

print("5 - NRMSE:", calc_nrmse(dec_img_100, dec_img_5), 
"SSIM:", calc_ssim(dec_img_100, dec_img_5),
"PSNR:", calc_psnr(dec_img_100, dec_img_5))

print("SP - NRMSE:", calc_nrmse(dec_img_100, dec_img_sp), 
"SSIM:", calc_ssim(dec_img_100, dec_img_sp),
"PSNR:", calc_psnr(dec_img_100, dec_img_sp))

stack = np.hstack([dec_img_100, dec_img_50, dec_img_25, dec_img_10, dec_img_5, dec_img_sp])

cv2.imshow('image',stack)
cv2.waitKey(0)
