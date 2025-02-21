import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.utils import plot_model  # type: ignore

# Ensure Graphviz is in the PATH for plot_model to work
os.environ["PATH"] += os.pathsep + r"C:\Users\K.lohith\Graphviz\Graphviz-12.2.1-win64\bin"


def load_and_preprocess_image(image_path):
    """Load and preprocess the image for the model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


def get_feature_maps(model, image):
    """Get feature maps from the model for the given image."""
    # Ensure the model is built before extracting feature maps
    _ = model.predict(np.zeros((1, 64, 64, 1)))  # Force model initialization

    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
    feature_map_model = Model(inputs=model.inputs, outputs=layer_outputs)
    feature_maps = feature_map_model.predict(image)
    return feature_maps


def plot_feature_maps(feature_maps, save_dir):
    """Plot and save the feature maps."""
    for layer_idx, feature_map in enumerate(feature_maps):
        num_filters = feature_map.shape[-1]

        # Create a grid of subplots dynamically
        plt.figure(figsize=(15, 15))
        cols = min(8, num_filters)  # Max 8 columns
        rows = (num_filters + cols - 1) // cols  # Auto-adjust rows

        for i in range(num_filters):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(feature_map[0, :, :, i], cmap='viridis')
            plt.axis('off')

        plt.suptitle(f'Feature Maps from Layer {layer_idx + 1}', fontsize=16)

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'feature_maps_layer_{layer_idx + 1}.png'))
        plt.close()  # Free memory


def save_cnn_architecture(model, save_path='cnn_architecture.png'):
    """Save the CNN model architecture as an image."""
    try:
        plot_model(model, to_file=save_path, show_shapes=True, show_layer_names=True)
        print(f"✅ Model architecture saved as {save_path}")
    except Exception as e:
        print(f"❌ Error saving model architecture: {e}")


def main():
    # Load the trained model
    model_path = 'asl_classifier.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    model = load_model(model_path)

    # Save CNN architecture
    save_cnn_architecture(model, 'cnn_architecture.png')

    # Specify the path to an image from your dataset
    dataset_path = 'dataset'
    letter = 'A'  # Change this as needed
    img_num = 0  # Change this as needed
    image_path = os.path.join(dataset_path, f"{letter}-samples", f"{img_num}.jpg")

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Create a directory to save feature maps
    save_dir = 'feature_maps'
    os.makedirs(save_dir, exist_ok=True)

    # Get feature maps
    feature_maps = get_feature_maps(model, image)

    # Plot and save feature maps
    plot_feature_maps(feature_maps, save_dir)


if __name__ == "__main__":
    main()
