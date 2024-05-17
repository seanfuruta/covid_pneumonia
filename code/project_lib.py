import matplotlib.pyplot as plt 
import numpy as np
from scipy.ndimage import convolve
from glob import glob
import os
import random
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import imageio.v3 as iio
import cv2 as cv
from scipy import signal
from scipy.ndimage import convolve

from matplotlib.patches import Ellipse
import skimage
from skimage import feature
from sklearn.mixture import GaussianMixture

from functools import reduce

data_path = "../data"
data_splits= ['Train', 'Val', 'Test']
classes = {'Normal':0, 'COVID-19':1, 'Non-COVID':2}
csv_path = "../metadata.csv"

val_data_file = '../feature_data/val_data.npy'
test_data_file = '../feature_data/test_data.npy'
train_data_file = '../feature_data/train_data.npy'

data_file = '../feature_data/all_data.npy'

cnn_data_file = '../feature_data/cnn_data.npy'

def generate_metadata_csv():
    if os.path.exists(csv_path):
        print("The Metadata file exist; cleaning")
        os.remove(csv_path)
    else:
        print("The Metadata file does not exist")
    
    print("Creating metadata")
    with open(csv_path, 'a') as csv_file:
        write = csv.writer(csv_file)
        write.writerow(['name', 'path', 'mask_path', 'class'])
        for c in classes.keys():
            files = glob(f'{data_path}/Train/{c}/images/*.png')
            rows = [[path.split('/')[-1], path, '/'.join(['/'.join(path.split('/')[:-2]), 'lung masks', path.split('/')[-1]]) , classes[c]] for path in files]
            write.writerows(rows)

def get_metadata():
    return pd.read_csv(csv_path)

def get_train_test_val_split(data):
    train, test = train_test_split(data, test_size=0.25)

    test, val = train_test_split(test, test_size=0.20)

    return (train, test, val)
            
def fetch_images(data):
    img_arr = data['path'].apply(lambda path: GenericWrapper(iio.imread(path))).to_numpy()
    return img_arr
    
def do_lung_mask_images(data, img_arr):
    masks = data['mask_path'].apply(lambda path: iio.imread(path)).to_numpy()

    img_arr_unwrapped = np.stack([img.get_data() for img in img_arr], axis=0)

    masks = np.concatenate(masks).reshape(-1, 256, 256)
    masked_img_arr = img_arr_unwrapped * masks
    
    return np.array([GenericWrapper(img) for img in masked_img_arr])

def fetch_lung_masked_images(data):
    img_arr = fetch_images(data)
    return do_lung_mask_images(data, img_arr)
    
def plot_images(imgs, title, wrapped=True):
    num = len(imgs)
    
    if num == 0:
        return
    elif num == 1:
        plt.figure(figsize=(6, 6))
        plt.imshow(imgs[0].get_data(), cmap='gray')
    else:
        _, ax = plt.subplots(1, num, figsize=((6 * num), 6))
        
        for i in range(num):

            if wrapped:
                ax[i].imshow(imgs[i].get_data(), cmap='gray')
            else:
                ax[i].imshow(imgs[i], cmap='gray')

    plt.suptitle(title, fontsize=20, color='gray')
    plt.show()

""" create a 2-D gaussian blurr filter for a given mean and std """
def create_2d_gaussian(size=9, std=1.5):
    gaussian_1d = signal.gaussian(size,std=std)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d/(gaussian_2d.sum())
    return gaussian_2d


""" normalize teh image between 0 and 1 """
def normalize_img(img):
    normalized = (img - img.min())/(img.max() - img.min())    
    return normalized

def local_normalize_img(img):
    float_gray = img.astype(np.float32) / 255.0

    blur = cv.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur

    blur = cv.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv.pow(blur, 0.5)

    gray = num / den

    cv.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
    return (gray * 255).astype(np.uint8)

### Laplacian and Gaussian stacks
# stack visualization
def visualize_stack(in_stack, title):
    # set the number of levels
    # create multi-row figure
    # add images to the figure from the stack
    # don't forget to set the cmap, vmin, vmax, and axis off
    # add titles to the plot
    levels = in_stack.shape[0]

    fig, ax = plt.subplots(nrows=1, ncols=levels, figsize=(3*levels, 3))

    for i in range(levels):
      ax[i].imshow(normalize_img(in_stack[i]), cmap='gray', vmin=0, vmax=1)
      ax[i].axis('off')
      ax[i].set_xticks([])
      ax[i].set_yticks([])

    plt.suptitle(title)
    plt.show()
    
# takes in a single channel
def gaussian_and_laplacian_stack(img, levels):
  # create 2D Gaussian
  # for each level
  # apply Gaussian and append to Gaussian stack
  # subtract current from previous Gaussian in stack
  # append result to Laplacian stack
  gs_h_arr = []
  ls_h_arr = []
  gs_h_arr.append(img)

  # gaussian kernel
  h = create_2d_gaussian(size=10, std=3)

  for i in range(levels-1):
    I_b = convolve(gs_h_arr[i], h, mode='wrap')
    gs_h_arr.append(I_b)
    l_diff = gs_h_arr[i] - I_b
    ls_h_arr.append(l_diff)

  # append last item to laplacian stack
  ls_h_arr.append(gs_h_arr[-1])

  gs_h = np.stack(gs_h_arr)
  ls_h = np.stack(ls_h_arr)
  return (gs_h, ls_h)

def get_blob_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = False
    params.minArea = 15

    # Filter by Circularity
    params.filterByCircularity = False
    #params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    # params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
      
    detector = cv.SimpleBlobDetector_create(params)
    
    return detector

class GenericWrapper:
    def __init__(self, data):
        self.data = data
    
    def get_data(self):
        return self.data

class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp

def report_clusterparams(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    U, s, Vt = np.linalg.svd(covariance)
    
    #print("U decomposition shape", U.shape)
    #print("S Decomposition shape", s.shape)
    #angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    #width, height = 2 * np.sqrt(s)
    
    
    # Draw the Ellipse
    #for nsig in range(1, 4):
        #ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
        
def plot_gmm(X, labels, label=True, ax=None):
    ax = ax or plt.gca()
    
    if label:
        ax.scatter(X['x'], X['y'], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X['x'], X['y'], s=40, zorder=2)
    ax.axis('equal')
    
def do_minmax_normalize(data):
    return (data-data.min())/(data.max()-data.min())
    
def get_preprocessed_img(img):
    clahe = cv.createCLAHE()
    denoising_h = 2
    unwrap_img = img.get_data()
    
    clahe_enhanced = clahe.apply(unwrap_img)
    local_norm_clahe = local_normalize_img(clahe_enhanced)
    denoised_clahe_enhanced = cv.fastNlMeansDenoising(local_norm_clahe, h=denoising_h)
    
    return GenericWrapper(denoised_clahe_enhanced)

def get_SIFT_keypoints_for_img(img):
    unwrap_img = img.get_data()
    sift = cv.SIFT_create()

    kp, _ = sift.detectAndCompute(unwrap_img, None)

    im_with_keypoints = cv.drawKeypoints(unwrap_img, kp, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return (kp, GenericWrapper(im_with_keypoints))

def get_sift_clusters(n_clusters=10, keypoints=None, plot=False):
    keypoint_arr = [[keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle] for keypoint in keypoints]
    
    keypoint_arr = pd.DataFrame(keypoint_arr, columns = ['x','y','size','angle']) 

    norm_keyarr = do_minmax_normalize(keypoint_arr)
    
    gmm = GaussianMixture(n_components = n_clusters)
    labels = gmm.fit(norm_keyarr).predict(norm_keyarr)
    
    if plot:
        plot_gmm(norm_keyarr, labels)
    
    norm_keyarr['label'] = labels
    
    cluster_prop_means = norm_keyarr.groupby('label', group_keys=True).mean().to_numpy().flatten()
    cluster_prop_stds = norm_keyarr.groupby('label', group_keys=True).std().to_numpy().flatten()
    cluster_prop_vars = norm_keyarr.groupby('label', group_keys=True).var().to_numpy().flatten()

    blob_clusters = np.concatenate((cluster_prop_means, cluster_prop_stds, cluster_prop_vars), axis=None)
    
    return blob_clusters

def get_blob_features(img):
    keypoints, kp_img = get_SIFT_keypoints_for_img(img)

    blob_features = get_sift_clusters(n_clusters=10, keypoints=keypoints, plot=False)
    
    return blob_features

def get_glcm_features(img):
    # Param:
    # source image
    # List of pixel pair distance offsets - here 1 in each direction
    # List of pixel pair angles in radians
    unwrap_img = img.get_data()
    graycom = skimage.feature.graycomatrix(unwrap_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    # Find the GLCM properties
    contrast = do_minmax_normalize(skimage.feature.graycoprops(graycom, 'contrast').flatten())
    dissimilarity = do_minmax_normalize(skimage.feature.graycoprops(graycom, 'dissimilarity').flatten())
    homogeneity = do_minmax_normalize(skimage.feature.graycoprops(graycom, 'homogeneity').flatten())
    energy = do_minmax_normalize(skimage.feature.graycoprops(graycom, 'energy').flatten())
    correlation = do_minmax_normalize(skimage.feature.graycoprops(graycom, 'correlation').flatten())
    ASM = do_minmax_normalize(skimage.feature.graycoprops(graycom, 'ASM').flatten())

    glcm_features = np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation, ASM), axis=None)
    
    return glcm_features

def get_lbp_features(img):
    unwrap_img = img.get_data()
    desc = LocalBinaryPatterns(24, 8)

    lbp_features, lbp = desc.describe(unwrap_img)

    lbp_features = do_minmax_normalize(lbp_features)
    
    return (lbp_features, GenericWrapper(lbp))

def get_X_y_for_training(data, save_to_file=True, filename=None):
    imgs = fetch_images(data)

    v_img_prep = np.vectorize(get_preprocessed_img)

    img_prepped = v_img_prep(imgs)

    img_prepped_masked = do_lung_mask_images(data, img_prepped)

    model_features = [np.concatenate([get_blob_features(img), get_glcm_features(img), get_lbp_features(img)[0]]) for img in img_prepped_masked]

    X = np.array(model_features, dtype=object)
    y = data['class'].to_numpy()

    if save_to_file:
        if os.path.exists(filename):
            print("Cleaning file, ", filename)
            os.remove(filename)
        else:
            print("No cleanup.. File doesn't exists")

        with open(filename, 'wb') as val_file:
            np.save(val_file, X)
            np.save(val_file, y)

    return (X,y)