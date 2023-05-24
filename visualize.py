import sys
sys.path.append("../")
from get_data_our.utils.image_utils import *
from get_data_our.model.res50sing_42 import *
from addon_vis.config import *
from addon_vis.attn_model import *
from addon_vis.plot import *
import os
import matplotlib.pyplot as plt
from PIL import Image

# Get the file name
file_name = input_file.split("/")[-1].split(".")[0]

# Get the image from the path
os.makedirs(output_dir, exist_ok=True)
print(f"Processing slide {input_file}")
all_patches, coordinates = image2patch(input_file, zoom_level, patch_size, output_dir, save_images)
print(f"Found {len(all_patches)} patches")

# split the patches in bg_patches and wsi_patches by saturation, save the corresponding coordinates
wsi_patches = []
bg_patches = []
wsi_coordinates = []
bg_coordinates = []

for patch, coordinate in zip(all_patches, coordinates):
    if saturation(patch) == True:
        wsi_patches.append(patch)
        wsi_coordinates.append(coordinate)
    else:
        bg_patches.append(patch)
        bg_coordinates.append(coordinate)

# Extract features from the patches with the pretrained model
features = extract_features(wsi_patches)
print(f"{len(wsi_patches)} contain tissue, {len(bg_patches)} do not contain tissue")

# Get the predictions from the loaded features
pred_class, pred_prob  = get_predictions(features, model_path)

# Load prediction dictionary from csv
prediction_dict = load_prediction_dict(prediction_dict_path)

# Print the correspodning class from the dictionary
predicted_class = None
for key, value in prediction_dict.items():
    if value == pred_class:
        predicted_class = key
        break

# Extract the attention from the features
attn = get_attention(features, model_path)
print(f"Extracted attention from {len(attn)} patches")
#for i in range(len(attn)):
#    print(f"Attention of patch {i}: {attn[i]}, coordinates: {wsi_coordinates[i]}")

# Visualize the attention using the coordinates of the patches, if the coordinates are not given, the patch is 0
attn = plot_attn(attn, wsi_coordinates, coordinates)
#plt.imsave(output_dir + file_name + "_attn1.jpg", attn)

# Read the image
ver = int(abs(math.log(float(zoom_level[:-1])/40, 2)))
image = read_slide(input_file, ver)

im = Image.fromarray(image)
im.save(output_dir + file_name + ".jpg")

# Scale the attention to the size of the image
sc_attn = scale_attn(attn, image.shape[0], image.shape[1], patch_size)
print(f"Rescaled attention to the size of the image: {sc_attn.shape}")
# Save the attention as an image
# plt.imsave(output_dir + file_name + "_attn2.jpg", sc_attn)

# Overlay the attention on the image
fig, ax = plt.subplots(figsize=(20,10))
im1 = ax.imshow(image)
cmap = "viridis"
im2 = ax.imshow(sc_attn, alpha=0.65, cmap=cmap)
cbar = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Attention', rotation=270, fontsize=20, labelpad=20)
plt.axis('off')

# Save the image
plt.savefig(output_dir + file_name + "_attn.jpg")

# Prediction
print(f"Predicted class: {predicted_class}, with probability {pred_prob*100:.2f}%")

# Save the class and the probability in a csv file, also the input file. 
# If the file already exists, append the new data
# Columns: file_name, predicted_class, probability
if os.path.isfile(output_dir + "predictions.csv"):
    with open(output_dir + "predictions.csv", "a") as f:
        f.write(f"{file_name},{predicted_class},{pred_prob}\n")
else:
    with open(output_dir + "predictions.csv", "w") as f:
        f.write("file_name,predicted_class,probability\n")
        f.write(f"{file_name},{predicted_class},{pred_prob}\n")