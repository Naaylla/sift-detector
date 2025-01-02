import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

query_images_path = "./dataset/testing"
descriptors_file = "./descriptors/descriptors.npy"

loaded_data = np.load(descriptors_file, allow_pickle=True)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

print("Available query images:")
query_images = os.listdir(query_images_path)
for idx, img_name in enumerate(query_images):
    print(f"{idx}: {img_name}")

choice = int(input("Select an image by its index: "))

query_image_name = query_images[choice]
query_image_path = os.path.join(query_images_path, query_image_name)

query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

if query_img is None:
    print(f"Error loading image: {query_image_path}")
    exit()

kp_query, des_query = sift.detectAndCompute(query_img, None)

best_match = None
best_image = None
best_score = float('inf')  

for entry in loaded_data:
    saved_filename = entry["filename"]
    saved_descriptors = entry["descriptors"]

    matches = bf.match(des_query, saved_descriptors)

    matches = sorted(matches, key=lambda val: val.distance)
    total_distance = sum(m.distance for m in matches[:50]) 

    if total_distance < best_score:
        best_score = total_distance
        best_match = matches
        best_image = saved_filename

if best_match:
    print(f"Best match found: {best_image} with a score of {best_score}")
    best_image_path = os.path.join("./dataset/training", best_image)
    best_img = cv2.imread(best_image_path, cv2.IMREAD_GRAYSCALE)

    img_matches = cv2.drawMatches(query_img, kp_query, best_img, sift.detectAndCompute(best_img, None)[0], best_match[:4], None, flags=2)
    plt.imshow(img_matches)
    plt.title("Query Image vs. Best Match")
    plt.show()
else:
    print("No suitable matches found.")
