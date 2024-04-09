'''
A script for importing saved data and producing the required
figures and plots
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2

##### Reconstruction Error Plots #####
error_ksvd_filename = "../data/cute_bear_ksvd_errors.npy"
error_sbl_filename = "../data/cute_bear_sbl_errors.npy"
noise_std_devs = np.array([15])

with open(error_ksvd_filename, 'rb') as f:
    errors_ksvd = np.load(f)
with open(error_sbl_filename, 'rb') as f:
    errors_sbl = np.load(f)

fig1, ax1 = plt.subplots()
ax1.plot(noise_std_devs, errors_ksvd, marker="*")
ax1.plot(noise_std_devs, errors_sbl, marker="*")
ax1.legend(["KSVD", "SBL"])
ax1.set_title("Image Reconstruction Error")
ax1.set_ylabel("noise standard deviation")

##### Reconstruction Error Plots #####
rt_ksvd_filename = "../data/cute_bear_ksvd_errors.npy"
rt_sbl_filename = "../data/cute_bear_sbl_errors.npy"

with open(rt_ksvd_filename, 'rb') as f:
    rt_ksvd = np.load(f)
with open(rt_sbl_filename, 'rb') as f:
    rt_sbl = np.load(f)

fig2, ax2 = plt.subplots()
ax2.plot(noise_std_devs, rt_ksvd, marker="*")
ax2.plot(noise_std_devs, rt_sbl, marker="*")
ax2.legend(["KSVD", "SBL"])
ax2.set_title("Algorithm Runtime")
ax2.set_ylabel("time [s]")
ax2.set_ylabel("noise standard deviation")

##### Print dictionary #####
method = "ksvd"
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
    block = vector.reshape((tile_size,tile_size), order="F")
    img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] = block
    col_idx = col_idx+tile_size
    if col_idx>=img_col:
        row_idx = row_idx+tile_size
        col_idx = 0
min_value = np.min(img)
img = img-min_value
max_value = np.max(abs(img))
img /= max_value
img *=255
img = np.clip(img, 0, 255).astype(np.uint8)

dict_filename = "../data/cute_bear_dict_"+method+tol+".jpg"
cv2.imwrite(dict_filename, img)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.show()