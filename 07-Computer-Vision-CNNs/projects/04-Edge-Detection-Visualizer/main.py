import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Running Project 04: Edge Detection Visualizer...")
    
    np.random.seed(42)
    # Generate simple test image (64x64) representing a square on a background
    image = np.zeros((64, 64))
    image[16:48, 16:48] = 255.0
    # Add minor noise
    image += np.random.normal(0, 5.0, image.shape)
    image = np.clip(image, 0.0, 255.0)
    
    # Sobel Kernels
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    h, w = image.shape
    grad_x = np.zeros((h - 2, w - 2))
    grad_y = np.zeros((h - 2, w - 2))
    
    # Manual Convolution
    for i in range(h - 2):
        for j in range(w - 2):
            patch = image[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(patch * Gx)
            grad_y[i, j] = np.sum(patch * Gy)
            
    # Calculate Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(grad_x, cmap='gray')
    axes[1].set_title("Horizontal Gradients (Gx)", fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(magnitude, cmap='gray')
    axes[2].set_title("Gradient Magnitude", fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plot_path = "edge_detection_output.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Visualization plot saved to '{plot_path}'")
    plt.close()
    
    print("Project 04 finished successfully.")

if __name__ == "__main__":
    main()
