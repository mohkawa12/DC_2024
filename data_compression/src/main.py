import cv2
import numpy as np
import random as rd
from ksvd import KSVD
from sbl import run_sbl_am
import time

####### Configuration #######
RUN_SBL = True
RUN_KSVD = False
LEARN_DICT = True
noise_std_devs = [5, 10]

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
    gauss_noise = np.random.normal(0, sigma, size=img.shape)
    out_img = img.astype(np.float64)+gauss_noise
    return np.clip(out_img, 0, 255).astype(np.uint8)

'''
Return noised images with std devs 5, 10, 15, 25
'''
def add_gauss_noise_set(img, std_devs):
    imgs = []
    for std_dev in std_devs:
        img = add_gauss_noise(img, std_dev)
        imgs.append(img)
        img_filename = "../data/cute_bear_ns"+str(std_dev)+".jpg"
        cv2.imwrite(img_filename, img)
    return imgs
        

img1_ns = add_gauss_noise_set(img1, noise_std_devs)

# To compare the original and noised images
# img1_concat = np.concatenate((img1, img1_ns[0], img1_ns[-1]), axis=1)
# cv2.imshow("image", img1_concat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####### Retrieve measurement vectors #######

'''
Split an image into 8x8 tiles, return the tiles as vectorised elements of a list.
'''
def get_tiles(img, overlap=False):
    if overlap==True:
        step = 2
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
        img_row = 320
        img_col = 480
        no_tiles = np.zeros((img_row, img_col))
        img = np.zeros((img_row, img_col))
        for vector in tiles:
            block = vector.reshape((tile_size,tile_size), order="F")
            img[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] += block
            no_tiles[row_idx:row_idx+tile_size, col_idx:col_idx+tile_size] += np.ones((tile_size, tile_size))
            col_idx = col_idx+2
            if col_idx+tile_size>img_col:
                row_idx = row_idx+2
                col_idx = 0
        img = img/no_tiles
    return np.clip(img, 0, 255).astype(np.uint8)

img1_ns_tiles = []
N=3000
yNs = []
for img1_noisy in img1_ns:
    img1_ns_tile = get_tiles(img1_noisy, overlap=True)
    img1_ns_tiles.append(img1_ns_tile)
    yNs.append(rd.sample(img1_ns_tile, N))

# For testing reconstruction
# img1_tiles = get_tiles(img1, overlap=True)
# img1 = get_image(img1_tiles, overlap=True)
# cv2.imshow("image", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if LEARN_DICT:
    # List of learned dictionaries
    As = []
    run_times = []
    dict_size = 250
    if RUN_KSVD:
    ######## KSVD #######
        method = "ksvd"
        for idx,yN in enumerate(yNs):
            start_time = time.time()
            tol = noise_std_devs[idx]  # or error tolerance
            ksvd = KSVD()
            A = ksvd.init_dict(dict_size)
            maxiter = 100
            old_avg_sparsity = dict_size # Initialise with full vector sparsity
            for i in range(maxiter):
                print("Finding sparse representation...")
                xK, new_avg_sparsity = ksvd.sparse_coding(A, np.array(yN).T, tol=tol)
                print("Updating dictionary...")
                A = ksvd.codebook_update(xK, yN, A)
                error = ksvd.convergence_crit(yN, A, xK)
                # print("Error after dictionary update is: ",error)
                if (abs(old_avg_sparsity-new_avg_sparsity)<0.01):
                    print("Sparsity level converged.")
                    break
                old_avg_sparsity = new_avg_sparsity
            As.append(A)
            run_time = time.time()-start_time
            run_times.append(run_time)
    elif RUN_SBL:
    ######## SBL #######
        method = "sbl"
        for idx,yN in enumerate(yNs):
            start_time = time.time()
            mu, A = run_sbl_am(sigma2=noise_std_devs[idx], Y=yN, num_atoms=dict_size)
            run_time = time.time()-start_time
            run_times.append(run_time)
            As.append(A)
    runtime_filename = "../data/cute_bear_"+method+"_rt.npy"
    # Save the runtimes and dictionaries 
    with open(runtime_filename, 'wb') as f:
        np.save(f, run_times)
    for idx, img1_noisy in enumerate(img1_ns):
        tol = noise_std_devs[idx] 
        dict_filename = "../data/cute_bear_dict_"+method+str(tol)+".npy"
        with open(dict_filename, 'wb') as f:
            np.save(f, As[idx])
else:
    if RUN_KSVD:
        method = "ksvd"
    elif RUN_SBL:
        method = "sbl"
    As = []
    for tol in noise_std_devs:
        dict_filename = "../data/cute_bear_dict_"+method+str(tol)+".npy"
        with open(dict_filename, 'rb') as f:
            A = np.load(f)
        As.append(A)

# Denoise the images using the learned dictionary
image_errors = []
print("Denonising images...")
ksvd = KSVD()
if RUN_KSVD:
    method = "ksvd"
elif RUN_SBL:
    method = "sbl"
for idx, img1_noisy in enumerate(img1_ns):
    tol = noise_std_devs[idx] 
    img1_tiles = get_tiles(img1_noisy, overlap=True)
    xK,_ = ksvd.sparse_coding(As[idx],np.array(img1_tiles).T, tol=tol)
    img1_dns_tiles = As[idx]@xK
    img1_dns = get_image(np.transpose(img1_dns_tiles), overlap=True)
    img_filename = "../data/cute_bear_"+method+str(tol)+".jpg"
    cv2.imwrite(img_filename, img1_dns)
