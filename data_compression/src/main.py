import cv2
import numpy as np
import random as rd
from ksvd import KSVD

####### Import Images #######
img1 = cv2.imread("../data/cute_bear.jpg", cv2.IMREAD_GRAYSCALE)

# To display the image
# cv2.imshow("image", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(img1.shape)

####### Add Gaussian Noise #######
'''
Add gaussian noise to img with std dev sigma
'''
def add_gauss_noise(img, sigma):
    gauss_noise = img.copy()
    cv2.randn(gauss_noise, 0, sigma)
    return cv2.add(img,gauss_noise)

'''
Return noised images with std devs 5, 10, 15, 25
'''
def add_gauss_noise_set(img):
    std_devs = [5, 10, 15, 25]
    imgs = []
    for std_dev in std_devs:
        imgs.append(add_gauss_noise(img, std_dev))
    return imgs
        

img1_ns = add_gauss_noise_set(img1)

# To compare the original and noised images
# img1_concat = np.concatenate((img1, img1_ns[1], img1_ns[-1]), axis=1)
# cv2.imshow("image", img1_concat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####### Retrieve measurement vectors #######

'''
Split an image into 8x8 tiles, return the tiles as vectorised elements of a list.
'''
def get_tiles(img):
    no_rows, no_cols = img.shape
    tile_size = 8
    all_tiles = []
    for row_idx in range(0, no_rows, tile_size):
        for col_idx in range(0, no_cols, tile_size):
            # Check to make sure we are not out of bounds of the image
            if (row_idx+tile_size>no_rows) or (col_idx+tile_size>no_cols):
                break
            # Retrieve the 8x8 tile
            tile = img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size]
            # Vectorise the tile (stack columns), append to list of vectorised tiles
            all_tiles.append(np.array(tile).ravel(order="F"))
    return all_tiles

img1_ns_tiles = get_tiles(img1_ns[0])
print(len(img1_ns_tiles))

N=500 # TODO should be 6000 once more images are included
yN = rd.sample(img1_ns_tiles, N)

######## KSVD #######
ksvd = KSVD()
A = ksvd.init_dict(100)
maxiter = 50
for i in range(maxiter):
    xK = ksvd.sparse_coding(A, yN, s=50)
    A = ksvd.codebook_update(xK, yN, A)