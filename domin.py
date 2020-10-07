import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(1)
def find_histogram(clt):

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


while True:
    success, box = cap.read()
    if success:
        img = box[210:270,210:370].copy()
        img = img.reshape((img.shape[0] * img.shape[1],3)) # represent as row*column, channel number
        clt = KMeans(n_clusters=3) # cluster number
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)
        box = cv2.rectangle(box,(210,210),(370,270),(0,0,255),3)
        
        cv2.imshow('dominant color',bar)
        cv2.imshow('box',box)
        cv2.imshow('img',img)
    else:
        print("FAIL")

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()