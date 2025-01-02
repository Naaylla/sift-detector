import numpy as np
import cv2
import os

model_images_path = "./dataset/training"
descriptors_folder = "./descriptors"

os.makedirs(descriptors_folder, exist_ok=True)

sift = cv2.SIFT_create()

data = []

for image in os.listdir(model_images_path):
    image_path = os.path.join(model_images_path, image)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to load {image_path}")
        continue
    
    kp, des = sift.detectAndCompute(img, None)
    
    data.append({
        "filename": image,
        "descriptors": des
    })
    print(f"Processed and saved descriptors for {image}")


descriptors_file = os.path.join(descriptors_folder, "descriptors.npy")
np.save(descriptors_file, data)
print(f"Saved all descriptors and filenames to {descriptors_file}")
