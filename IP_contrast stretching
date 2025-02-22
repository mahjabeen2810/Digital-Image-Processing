import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("sunflower.jpg",0)
min_intensity=np.min(img)
max_intensity=np.max(img)

stretched_image=((img-min_intensity)/(max_intensity-min_intensity)*255).astype(np.uint8)

hist_original=cv2.calcHist([img],[0],None,[255],[0,255])
hist_stretched=cv2.calcHist([stretched_image],[0],None,[255],[0,255])

equalized_image=cv2.equalizeHist(img)
hist_equalize=cv2.calchist([equalized_image],[0],None,[255],[0,255])

plt.figure(figsize=(12,8))

plt.subplot(231)
plt.title("Original Image")
plt.imshow(img,"Accent")
plt.axis=("off")

plt.subplot(232)
plt.title("Stretched Image")
plt.imshow(stretched_image,"Accent")
plt.axis=("off")

plt.subplot(233)
plt.title("Equalized Image")
plt.imshow(equalized_image,"Accent")
plt.axis=("off")

plt.subplot(234)
plt.title("Histogram original")
plt.plot(hist_original)
plt.xlim([0,255])

plt.subplot(235)
plt.title("Histogram Stretched")
plt.plot(hist_stretched)
plt.xlim([0,255])

plt.subplot(236)
plt.title("Histogram Equalized")
plt.plot(hist_equalize)
plt.xlim([0,255])