from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#This coding used below link as the reference
#credit to: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
#credit to: https://www.youtube.com/watch?v=1FJWXOO1SRI&t=939s
#credit to: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
#The commented codes were used to find out the dominated color

cap = cv2.VideoCapture(1)
tracker = cv2.TrackerMOSSE_create()
works, img = cap.read()
ctp = cv2.selectROI("Tracking",img,False)
tracker.init(img,ctp)

def contours(img,ctp):
    x,y,w,h = int(ctp[0]),int(ctp[1]),int(ctp[2]),int(ctp[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    
# def find_histogram(clt):
    # """
    # create a histogram with k clusters
    # :param: clt
    # :return:hist
    # """
    # numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    # (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # hist = hist.astype("float")
    # hist /= hist.sum()

    # return hist
# def plot_colors2(hist, centroids):
    # bar = np.zeros((50, 300, 3), dtype="uint8")
    # startX = 0

    # for (percent, color) in zip(hist, centroids):
        # # plot the relative percentage of each cluster
        # endX = startX + (percent * 300)
        # cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      # color.astype("uint8").tolist(), -1)
        # startX = endX

    # # return the bar chart
    # return bar
    

while True:


    works,img=cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # To RGB
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_2 = cv2.inRange(rgb, lower_blue, upper_blue)
    res = cv2.bitwise_and(img,img, mask= mask)
    
    works,ctp=tracker.update(img)
    print(ctp)
    #clt = KMeans(n_clusters=3) #cluster number
    #clt.fit(img)

    # hist = find_histogram(clt)
    # bar = plot_colors2(hist, clt.cluster_centers_)
    
    if works:
        contours(img,ctp)
    else:
        print("Nothing can capture")
        
        
    cv2.imshow("Tracking",img)
    cv2.imshow("mask",mask)
    cv2.imshow("mask_2",mask_2)
    cv2.imshow("mask",mask)
    #cv2.imshow('dominant color',bar)
 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
        
