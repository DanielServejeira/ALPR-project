import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def morphological_skeleton(image):
    img = image.copy()
    skel = np.zeros(img.shape, np.uint8)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv.erode(img, kernel)
        temp = cv.dilate(eroded, kernel)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv.countNonZero(img) == 0:
            break

    return skel

# Lê a imagem em BGR
img = cv.imread('./train/S/808.png', cv.IMREAD_GRAYSCALE)

img = cv.bitwise_not(img)

blur = cv.GaussianBlur(img,(5,5),0)

equalized = cv.equalizeHist(blur)

ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

cv.imwrite('./output.png', opening)  # OpenCV espera BGR par salvar
