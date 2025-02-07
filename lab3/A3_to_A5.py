import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
import matplotlib.pyplot as plt


def load_dataset(dataset_path):
    """
    Load images and labels from the dataset directory.
    Dataset structure: dataset/A-samples/0.jpg to dataset/Y-samples/99.jpg
    Returns:
        X: numpy array of images
        y: numpy array of labels
    """
    images = []
    labels = []
    label_mapping = {}

    # Create label mapping for letters A-Y (excluding J and Z)
    valid_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
    label_mapping = {letter: idx for idx, letter in enumerate(valid_letters)}

    # Walk through dataset directory
    for letter in valid_letters:
        folder_name = f"{letter}-samples"
        folder_path = os.path.join(dataset_path, folder_name)

        if os.path.isdir(folder_path):
            for img_name in range(100):  # 0.jpg to 99.jpg
                img_path = os.path.join(folder_path, f"{img_name}.jpg")
                if os.path.exists(img_path):
                    # Load and preprocess image
                    img = load_img(img_path, target_size=(64, 64), color_mode='grayscale')
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalize pixel values

                    images.append(img_array)
                    labels.append(label_mapping[letter])

    return np.array(images), np.array(labels), len(label_mapping)


def create_model(input_shape, num_classes):
    """
    Create a CNN model with better architecture for image classification.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation', marker='o')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation', marker='o')
    ax2.set_title('Model Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_model(model, dataset_path, label_mapping):
    """
    Test the model on a few sample images and visualize results.
    Args:
        model: Trained Keras model
        dataset_path: Path to dataset
        label_mapping: Dictionary mapping indices to letters
    """
    # Reverse the label mapping to get letters from indices
    idx_to_letter = {idx: letter for letter, idx in label_mapping.items()}

    # Create a figure with subplots
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()

    # Randomly select some test images
    letters = list(label_mapping.keys())
    np.random.shuffle(letters)
    test_letters = letters[:16]  # Test 16 random letters

    for idx, letter in enumerate(test_letters):
        # Load a random image for this letter
        folder_name = f"{letter}-samples"
        img_num = np.random.randint(0, 100)
        img_path = os.path.join(dataset_path, folder_name, f"{img_num}.jpg")

        # Load and preprocess image
        img = load_img(img_path, target_size=(64, 64), color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = img_array / 255.0

        # Get prediction
        prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
        predicted_idx = np.argmax(prediction[0])
        predicted_letter = idx_to_letter[predicted_idx]
        confidence = prediction[0][predicted_idx] * 100

        # Display image and prediction
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        color = 'green' if predicted_letter == letter else 'red'
        axes[idx].set_title(f'True: {letter}\nPred: {predicted_letter}\nConf: {confidence:.1f}%',
                            color=color)

    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()


def evaluate_model(model, X_test, y_test, label_mapping):
    """
    Evaluate model performance with detailed metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate accuracy per class
    class_accuracies = {}
    idx_to_letter = {idx: letter for letter, idx in label_mapping.items()}

    for idx in range(len(label_mapping)):
        mask = (y_test_classes == idx)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == idx)
            letter = idx_to_letter[idx]
            class_accuracies[letter] = class_acc * 100

    # Print detailed metrics
    print("\nDetailed Evaluation Metrics:")
    print("-" * 40)
    print("Per-Class Accuracies:")
    for letter, acc in sorted(class_accuracies.items()):
        print(f"Letter {letter}: {acc:.2f}%")

    # Overall metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("-" * 40)
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Overall Test Loss: {test_loss:.4f}")


def main():
    # Load and preprocess dataset
    dataset_path = 'dataset'
    X, y, num_classes = load_dataset(dataset_path)

    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create and compile model
    input_shape = (64, 64, 1)
    model = create_model(input_shape, num_classes)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Increased epochs
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Create label mapping
    valid_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
    label_mapping = {letter: idx for idx, letter in enumerate(valid_letters)}

    # Test model visualization
    test_model(model, dataset_path, label_mapping)

    # Detailed evaluation
    evaluate_model(model, X_test, y_test, label_mapping)

    # Save the model
    model.save('asl_classifier.keras')  # Using .keras format as recommended


if __name__ == "__main__":
    main()