import cv2
import numpy as np
import random as rd
from ksvd import KSVD
from sbl import run_sbl_am

####### Import Images #######
img1 = cv2.imread("../data/cute_bear.jpg", cv2.IMREAD_GRAYSCALE)

# To display the image
# cv2.imshow("image", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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
def get_tiles(img, overlap=False):
    if overlap==True:
        step = 4
    else:
        step = 8
    no_rows, no_cols = img.shape
    tile_size = 8
    all_tiles = []
    for row_idx in range(0, no_rows, step):
        for col_idx in range(0, no_cols, step):
            # Check to make sure we are not out of bounds of the image
            if (row_idx+tile_size>no_rows) or (col_idx+tile_size>no_cols):
                break
            # Retrieve the 8x8 tile
            tile = img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size]
            # Vectorise the tile (stack columns), append to list of vectorised tiles
            all_tiles.append(np.array(tile).ravel(order="F"))
    return all_tiles

'''
Return an image from the 8x8 tiles
'''
def get_image(tiles, overlap=False):
    img_row = 320
    img_col = 480
    img = np.zeros((img_row, img_col))
    tile_size = 8
    row_idx = 0
    col_idx = 0
    if overlap==False:
        for vector in tiles:
            block = vector.reshape((tile_size,tile_size), order="F")
            img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] = block
            col_idx = col_idx+tile_size
            if col_idx>=img_col:
                row_idx = row_idx+tile_size
                col_idx = 0
    else:
        img_row = 321
        img_col = 481
        no_tiles = np.zeros((img_row, img_col))
        img = np.zeros((img_row, img_col))
        for vector in tiles:
            block = vector.reshape((tile_size,tile_size), order="F")
            img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] += block
            no_tiles[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] += np.ones((tile_size, tile_size))
            col_idx = col_idx+4
            if col_idx+tile_size>img_col:
                row_idx = row_idx+4
                col_idx = 0
        img = img/no_tiles
    return img.astype(np.uint8)


img1_ns_tiles = get_tiles(img1_ns[0], overlap=True)
print(len(img1_ns_tiles))

# For testing reconstruction
# img1_tiles = get_tiles(img1, overlap=True)
# img1 = get_image(img1_tiles, overlap=True)
# cv2.imshow("image", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

N=500
yN = rd.sample(img1_ns_tiles, N)

######## KSVD #######
s = 50 # assumed sparsity
tol = 1 # or error tolerance
ksvd = KSVD()
A = ksvd.init_dict(200)
maxiter = 50
for i in range(maxiter):
    print("Finding sparse representation...")
    xK = ksvd.sparse_coding(A, np.array(yN).T, tol=tol)
    print("Updating dictionary...")
    A = ksvd.codebook_update(xK, yN, A)
    error = ksvd.convergence_crit(yN, A, xK)
    print("Error is: ",error)
    if (error<20):
        break

######## SBL #######
mu, A = run_sbl_am(sigma2=5, Y=yN, num_atoms=300)

# Denoise the image
img1_tiles = get_tiles(img1_ns[0], overlap=False)
xK = ksvd.sparse_coding(A,np.array(img1_tiles).T, tol=tol)
img1_dns_tiles = A@xK
img1_dns = get_image(np.transpose(img1_dns_tiles), overlap=False)
cv2.imshow("image", img1_dns)
cv2.waitKey(0)
cv2.destroyAllWindows()