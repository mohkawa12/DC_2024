'''
A script for importing saved data and producing the required
figures and plots
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2

noise_std_devs = np.array([5, 10, 15, 25])
##### Reconstruction Error Plots #####
# This is not a good measure of image reconstruction
# def image_error(orig_img, new_img):
#     row, col = new_img.shape
#     orig_img = orig_img[:row,:col] 
#     error = np.linalg.norm(new_img-orig_img)/(row*col)
#     return error

# row = 320
# col = 480
# original_img = cv2.imread("../data/cute_bear.jpg", cv2.IMREAD_GRAYSCALE)
# original_img = original_img[:row,:col]
# errors_ksvd = []
# errors_sbl = []
# for noise_level in noise_std_devs:
#     ksvdimg_filename = "../data/cute_bear_ksvd"+str(noise_level)+".jpg"
#     sblimg_filename = "../data/cute_bear_sbl"+str(noise_level)+".jpg"
#     ksvddns_img = cv2.imread(ksvdimg_filename, cv2.IMREAD_GRAYSCALE)
#     sbldns_img = cv2.imread(sblimg_filename, cv2.IMREAD_GRAYSCALE)
#     errors_ksvd.append(image_error(original_img, ksvddns_img))
#     errors_sbl.append(image_error(original_img, sbldns_img))

# fig1, ax1 = plt.subplots()
# ax1.plot(noise_std_devs, errors_ksvd, marker="*")
# ax1.plot(noise_std_devs, errors_sbl, marker="*")
# ax1.legend(["KSVD", "SBL"])
# ax1.set_title("Image Reconstruction Error")
# ax1.set_xlabel("noise standard deviation")
# ax1.grid()

##### Runtime Plots #####
# rt_ksvd_filename = "../data/cute_bear_ksvd_rt.npy"
# rt_sbl_filename = "../data/cute_bear_sbl_rt.npy"

# with open(rt_ksvd_filename, 'rb') as f:
#     rt_ksvd = np.load(f)
# with open(rt_sbl_filename, 'rb') as f:
#     rt_sbl = np.load(f)

# fig2, ax2 = plt.subplots()
# ax2.plot(noise_std_devs, rt_ksvd, marker="*")
# ax2.plot(noise_std_devs, rt_sbl, marker="*")
# ax2.legend(["KSVD", "SBL"])
# ax2.set_title("Algorithm Runtime")
# ax2.set_ylabel("time [s]")
# ax2.set_xlabel("noise standard deviation")
# ax2.grid()


##### Peak SNR #####
from math import log10, sqrt

def PSNR(original_img, new_img):
    error = np.mean((original_img-new_img)**2)
    return 20*log10(255/sqrt(error))

row = 320
col = 480
original_img = cv2.imread("../data/cute_bear.jpg", cv2.IMREAD_GRAYSCALE)
original_img = original_img[:row,:col]
ksvd_psnr = []
sbl_psnr = []
for noise_level in noise_std_devs:
    ksvdimg_filename = "../data/cute_bear_ksvd"+str(noise_level)+".jpg"
    sblimg_filename = "../data/cute_bear_sbl"+str(noise_level)+".jpg"
    ksvddns_img = cv2.imread(ksvdimg_filename, cv2.IMREAD_GRAYSCALE)
    sbldns_img = cv2.imread(sblimg_filename, cv2.IMREAD_GRAYSCALE)
    ksvd_psnr.append(PSNR(original_img, ksvddns_img))
    sbl_psnr.append(PSNR(original_img, sbldns_img))
    

fig3, ax3 = plt.subplots()
ax3.plot(noise_std_devs, ksvd_psnr, marker="*")
ax3.plot(noise_std_devs, sbl_psnr, marker="*")
ax3.legend(["KSVD", "SBL"])
ax3.set_title("Peak SNR")
ax3.set_ylabel("dB")
ax3.set_xlabel("noise standard deviation")
ax3.grid()

##### SSIM #####
from skimage.metrics import structural_similarity as ssim

row = 320
col = 480
original_img = cv2.imread("../data/cute_bear.jpg", cv2.IMREAD_GRAYSCALE)
original_img = original_img[:row,:col]
ksvd_ssim = []
sbl_ssim = []
for noise_level in noise_std_devs:
    ksvdimg_filename = "../data/cute_bear_ksvd"+str(noise_level)+".jpg"
    sblimg_filename = "../data/cute_bear_sbl"+str(noise_level)+".jpg"
    ksvddns_img = cv2.imread(ksvdimg_filename, cv2.IMREAD_GRAYSCALE)
    sbldns_img = cv2.imread(sblimg_filename, cv2.IMREAD_GRAYSCALE)
    ksvd_ssim.append(ssim(original_img, ksvddns_img, data_range=255))
    sbl_ssim.append(ssim(original_img, sbldns_img, data_range=255))

fig4, ax4 = plt.subplots()
ax4.plot(noise_std_devs, ksvd_ssim, marker="*")
ax4.plot(noise_std_devs, sbl_ssim, marker="*")
ax4.legend(["KSVD", "SBL"])
ax4.set_title("Structural Similarity Index")
# ax4.set_ylabel("dB")
ax4.set_xlabel("noise standard deviation")
ax4.grid()

##### SSIM #####


##### Print dictionary #####
method = "sbl"
tol = "15"
dict_filename = "../data/cute_bear_dict_"+method+tol+".npy"
with open(dict_filename, 'rb') as f:
    A = np.load(f)
img_row = int(A.shape[1]/25*8)
img_col = 200
img = np.zeros((img_row, img_col))
tile_size = 8
row_idx = 0
col_idx = 0
for vector in A.T:
    # Convert the l2 normalised vector to [0, 255]
    vector -= np.min(vector)
    vector /= np.max(vector)
    vector *= 255
    block = vector.reshape((tile_size,tile_size), order="F")
    img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] = block
    col_idx = col_idx+tile_size
    if col_idx>=img_col:
        row_idx = row_idx+tile_size
        col_idx = 0
# min_value = np.min(img)
# img = img-min_value
# max_value = np.max(abs(img))
# img /= max_value
# img *=255
img = np.clip(img, 0, 255).astype(np.uint8)

dict_filename = "../data/cute_bear_dict_"+method+tol+".jpg"
cv2.imwrite(dict_filename, img)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()