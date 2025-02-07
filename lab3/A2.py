import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.io import imread
from skimage.color import rgb2gray

# Load and preprocess the image
im = imread("Neural.JPG")
img = rgb2gray(im) * 255

# Crop the region of interest
img1 = img[40:350, 20:350]

# Define filters
fil1 = np.array([[ 0, -1,  0], [-1, 4, -1], [ 0, -1, 0]])  # Edge detection
fil2 = np.array([[ 0.2, 0.5,  0.2], [0.5, 1, 0.5], [0.2, 0.5, 0.2]])  # Sharpening
fil3 = np.full((5, 5), 0.1)  # Smoothing

# Apply convolution
grad1 = signal.convolve2d(img1, fil1, boundary='symm', mode='same')
grad2 = signal.convolve2d(img1, fil2, boundary='symm', mode='same')
grad3 = signal.convolve2d(img1, fil3, boundary='symm', mode='same')

# Display results
plt.figure(figsize=(12,4))
plt.subplot(1,4,1)
plt.imshow(img1, cmap='gray')
plt.title("Original Image")

plt.subplot(1,4,2)
plt.imshow(np.abs(grad1), cmap='gray')
plt.title("Edge Detection")

plt.subplot(1,4,3)
plt.imshow(grad2, cmap='gray')
plt.title("Sharpening")

plt.subplot(1,4,4)
plt.imshow(grad3, cmap='gray')
plt.title("Smoothing")

plt.show()
