import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels, num_images=10):
    plt.figure(figsize=(12,6))
    for i in range(num_images):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Labels: {labels[i]}")
        plt.axis('off')
    plt.show()

def show_image(image, label):
    plt.figure(figsize=(12,6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

