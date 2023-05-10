#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install image_dehazer


# # code for displaying the histogram of image

# In[51]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

# read in the image using OpenCV
img = cv2.imread("pout.tif")

# convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# display the image
plt.imshow(img_gray, cmap='gray')

# plot the histogram
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.figure()
plt.title("Histogram of Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


# # Adjusting Image Histograms to Increase Image Contrast

# In[42]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("pout.tif", cv2.IMREAD_GRAYSCALE)

img_equalized = cv2.equalizeHist(img)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].hist(img.ravel(), bins=256)
ax[1].set_xlim(0, 255)
ax[1].set_ylim(0, 4000)
ax[1].set_title('Original histogram')

ax[2].imshow(img_equalized, cmap=plt.cm.gray)
ax[2].set_title('Equalized image')

ax[3].hist(img_equalized.ravel(), bins=256)
ax[3].set_xlim(0, 255)
ax[3].set_ylim(0, 4000)
ax[3].set_title('Equalized histogram')

plt.tight_layout()
plt.show()


# 
# # Focusing on a Specific Brightness Range

# In[38]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
xrayImg = cv2.imread("armxray.png", cv2.IMREAD_GRAYSCALE)

# Display the original image and its histogram
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(xrayImg, cmap='gray')
ax0.set_title('Original Image')
ax1.hist(xrayImg.ravel(), bins=256, range=(0, 255))
ax1.set_title('Histogram')

# Rescale the intensity to only show pixel values between 225 and 255
xrayPins = cv2.convertScaleAbs(xrayImg, alpha=(255.0/(255-225)), beta=-225*(255.0/(255-225)))
fig, (ax2, ax3) = plt.subplots(ncols=2, figsize=(10, 5))
ax2.imshow(xrayImg, cmap='gray')
ax2.set_title('Original Image')
ax3.imshow(xrayPins, cmap='gray')
ax3.set_title('Rescaled Image (225-255)')
plt.show()


# # Histogram Matching

# In[52]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the reference and crack images in grayscale
refImg = cv2.imread("00001.jpg", cv2.IMREAD_GRAYSCALE)
crackImg = cv2.imread("00115.jpg", cv2.IMREAD_GRAYSCALE)

# Display the reference and crack images side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(refImg, cmap=plt.cm.gray)
axes[0].set_title('Reference Image')
axes[1].imshow(crackImg, cmap=plt.cm.gray)
axes[1].set_title('Crack Image')
plt.show()

# Calculate the histogram of the reference and crack images
refHist, _ = np.histogram(refImg.ravel(), bins=256, range=[0, 256])
crackHist, _ = np.histogram(crackImg.ravel(), bins=256, range=[0, 256])

# Plot the histograms of the reference and crack images
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].plot(refHist)
axes[0].set_xlim([0, 256])
axes[0].set_title('Reference Image Histogram')
axes[1].plot(crackHist)
axes[1].set_xlim([0, 256])
axes[1].set_title('Crack Image Histogram')
plt.show()

# Equalize the histogram of the crack image to match the reference image
matchImg = cv2.equalizeHist(crackImg)

# Calculate the histogram of the matched image
matchHist, _ = np.histogram(matchImg.ravel(), bins=256, range=[0, 256])

# Plot the histogram of the matched image
plt.plot(matchHist)
plt.xlim([0, 256])
plt.title('Matched Image Histogram')
plt.show()

# Display the original and matched crack images side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(crackImg, cmap=plt.cm.gray)
axes[0].set_title('Original Crack Image')
axes[1].imshow(matchImg, cmap=plt.cm.gray)
axes[1].set_title('Matched Crack Image')
plt.show()


# # Adjusting HSV Color Planes

# In[53]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# read in the RGB image
img = cv2.imread("sherlock.jpg")

# convert RGB image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# plot the histograms of the individual channels
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].hist(hsv_img[:, :, 0].ravel(), bins=256, color='r')
axs[0].set_title('Hue')
axs[1].hist(hsv_img[:, :, 1].ravel(), bins=256, color='g')
axs[1].set_title('Saturation')
axs[2].hist(hsv_img[:, :, 2].ravel(), bins=256, color='b')
axs[2].set_title('Value')
plt.show()

# stretch the histogram of the value channel
hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])

# convert HSV image back to RGB color space
new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

# display the original and adjusted images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(122)
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.title('Adjusted Image')
plt.show()


# In[ ]:





# In[ ]:


import cv2
import numpy as np

# read in RGB image
img = cv2.imread("car_2.jpg")

# convert to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# perform histogram equalization on value plane
hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])

# convert back to RGB
new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

# display original and new images side by side
cv2.imshow("Original Image", img)
cv2.imshow("Adjusted Image", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




