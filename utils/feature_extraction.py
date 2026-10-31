import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops

IMG_SIZE = 128

def extract_features(image):

    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    hog_features = hog(gray, orientations=9, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), visualize=False)

    glcm = graycomatrix(gray,[1],[0],256,True,True)

    contrast = graycoprops(glcm,'contrast')[0][0]
    energy = graycoprops(glcm,'energy')[0][0]
    homogeneity = graycoprops(glcm,'homogeneity')[0][0]

    mean_color = np.mean(image,axis=(0,1))
    std_color = np.std(image,axis=(0,1))

    return np.hstack((hog_features,contrast,energy,homogeneity,mean_color,std_color))