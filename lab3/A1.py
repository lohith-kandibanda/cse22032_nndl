import numpy as np
import matplotlib.pyplot as plt

# Define signal and filters
X = [0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,0,0]
H_L = [0.05, 0.2, 0.5, 0.2, 0.05]  # Low-pass filter
H_H = [-1, 2, -1]  # High-pass filter

# Perform convolution
y_low = np.convolve(X, H_L, mode='same')
y_high = np.convolve(X, H_H, mode='same')

# Plot results
plt.figure(figsize=(10,5))
plt.plot(X, label='Original Signal', linestyle='dashed')
plt.plot(y_low, label='Low-Pass Filter Output')
plt.plot(y_high, label='High-Pass Filter Output')
plt.legend()
plt.show()
