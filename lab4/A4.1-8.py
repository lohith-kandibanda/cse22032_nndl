import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# ----------------------------- Load Dataset ----------------------------- #

def load_dataset(dataset_path, img_size=32):
    """
    Load images and labels from the dataset directory.
    Returns:
        X: numpy array of images
        y: numpy array of labels
        num_classes: total number of classes (A-Y, excluding J and Z)
    """
    images = []
    labels = []

    # Create label mapping for letters A-Y (excluding J and Z)
    valid_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
    label_mapping = {letter: idx for idx, letter in enumerate(valid_letters)}

    # Walk through dataset directory
    for letter in valid_letters:
        folder_name = f"{letter}-samples"
        folder_path = os.path.join(dataset_path, folder_name)

        if os.path.isdir(folder_path):
            for img_idx in range(100):  # 0.jpg to 99.jpg
                img_path = os.path.join(folder_path, f"{img_idx}.jpg")
                if os.path.exists(img_path):
                    # Load and preprocess image
                    img = load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')
                    img_array = img_to_array(img) / 255.0  # Normalize pixel values

                    images.append(img_array)
                    labels.append(label_mapping[letter])

    X = np.array(images)
    y = np.array(labels)
    num_classes = len(label_mapping)

    return X, y, num_classes


# Load dataset
dataset_path = "dataset"  # Change this to your actual dataset path
img_size = 32  # Reduce size to 32x32 for better FC model performance
X, y, num_classes = load_dataset(dataset_path, img_size)

# Split dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Dataset loaded successfully! {X_train.shape[0]} training images, {X_val.shape[0]} validation images, {X_test.shape[0]} test images.")

# ----------------------------- A1 & A2: CNN Model ----------------------------- #

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_val, y_val))

# ----------------------------- A2: Plot Training Curves ----------------------------- #

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# ----------------------------- A3: Evaluate CNN Model ----------------------------- #

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy (CNN): {test_acc:.4f}")

# ----------------------------- A4: Inspect Filters ----------------------------- #

filters, biases = model.layers[0].get_weights()
filters = (filters - filters.min()) / (filters.max() - filters.min())  # Normalize

# Plot first 5 filters
for i in range(5):
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.title(f'Filter {i+1}')
    plt.show()

# ----------------------------- A5: Apply Filter to Image ----------------------------- #

from scipy import signal

im = X_train[10].reshape(img_size, img_size)
selected_filter = filters[:, :, 0, 1]  # Selecting 2nd filter

output = signal.convolve2d(im, selected_filter.reshape(3, 3), mode='same')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Filtered Image")

plt.show()

# ----------------------------- A6: Fully Connected Model ----------------------------- #


from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore

# Flatten images
X_train_fc = X_train.reshape(-1, img_size * img_size)
X_val_fc = X_val.reshape(-1, img_size * img_size)
X_test_fc = X_test.reshape(-1, img_size * img_size)

# Define FC model with Batch Normalization
fc_model = Sequential([
    Dense(1024, activation='relu', input_shape=(img_size * img_size,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Learning rate scheduling
optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

fc_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model for 30 epochs
fc_history = fc_model.fit(X_train_fc, y_train, batch_size=32, epochs=30, validation_data=(X_val_fc, y_val), callbacks=[lr_scheduler])

# ----------------------------- A7: Training Curves (FC Model) ----------------------------- #

plt.plot(fc_history.history['loss'], label='Training Loss')
plt.plot(fc_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss (FC Model)")
plt.show()

# ----------------------------- A8: Evaluate Fully Connected Model ----------------------------- #

fc_score = fc_model.evaluate(X_test_fc, y_test, verbose=1)
print(f'Fully Connected Model Test Accuracy: {fc_score[1]:.4f}')
