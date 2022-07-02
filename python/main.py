# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

import numpy as np
from scipy import ndimage
from skimage import color
import cv2, scipy

font = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (10,25)
fontScale = 1
fontColor = (0,0,0)
thickness = 2
linetype = 2


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x,y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma ** 2))) * normal
    return g

def sobel_filter(img):

    sx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], np.float32)
    sy = np.array([[1,2,1], [0,0,0], [-1, -2, -1]], np.float32)

    ix = cv2.filter2D(src=img, ddepth=-1, kernel=sx)
    iy = cv2.filter2D(src=img, ddepth=-1, kernel=sy)

    # ix = ndimage.filters.convolve(img, sx)
    # iy = ndimage.filters.convolve(img, sy)

    # cv2.imshow('dd', ix)

    g = np.hypot(ix, iy)

    # e = np.concatenate((ix,iy,g), axis=1)
    # cv2.imshow('sobel filters', e)
    
    ### BROKEN? supposed to scale to 0-255, but breaks stuff if left in idk
    #g = g/g.max() * 255 #scale image from 0-255


    theta = np.arctan2(iy, ix)

    return g, theta

def non_max_supression(img, d):
    m,n = img.shape
    z = np.zeros((m,n))

    angle = d * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1,m-1):
        for j in range(1,n-1):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if (img[i,j] >= q) and (img[i,j] >= r):
                    z[i,j] = img[i,j]
                else:
                    z[i,j] = 0
            except IndexError as e:
                pass
    return z

def threshold(img, lowThresholdRatio=0.04, highThresholdRatio=0.15):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def main():
    a = 0
    print(a)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = color.rgb2gray(frame) #convert to greyscale
        frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation=cv2.INTER_AREA) #rescale image to half dimensions
        print(frame.shape)
        g = gaussian_kernel(5) #3x3 kernel
        applied_gauss = cv2.filter2D(src=frame, kernel=g, ddepth=-1)
        gradient, theta = sobel_filter(applied_gauss)
        nonMaxImg = non_max_supression(gradient, theta)
        (thresholdImage, weak, strong) = threshold(nonMaxImg)
        img_final = hysteresis(thresholdImage, weak, strong)

        # print(gradient, theta)
        cv2.putText(frame, "original image", topLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)
        cv2.putText(applied_gauss, "gauss filter", topLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)
        cv2.putText(gradient, "sobel filters", topLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)
        cv2.putText(nonMaxImg, "reduce line thinkness", topLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)
        cv2.putText(thresholdImage, "thresholding image", topLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)
        cv2.putText(img_final, "Final Image", topLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)
        
        show_all = np.concatenate((frame, applied_gauss, gradient, nonMaxImg, thresholdImage, img_final), axis=1)
        # show_all = np.concatenate(frame)
        cv2.imshow('Input', show_all)

        c = cv2.waitKey(1)
        if (c == 27):
            break
    
    cap.release()
    cv2.destroyAllWindows()



main()
