import os
from PIL import Image
from tqdm import tqdm
# Define the directory containing the .tiff images
input_directory = "/home/macula/SMATousi/Gullies/ground_truth/organized_data/All_Pos_Neg/combined_rgb_folder"
output_directory = "/home/macula/SMATousi/Gullies/ground_truth/organized_data/All_Pos_Neg/combined_rgb_folder_jpg"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each file in the input directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith(".tiff") or filename.endswith(".tif"):
        # Open the .tiff image
        tiff_path = os.path.join(input_directory, filename)
        with Image.open(tiff_path) as img:
            # Convert the image to RGB (if it's not already)
            rgb_img = img.convert("RGB")
            # Define the output path with .jpeg extension
            jpeg_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.jpg")
            # Save the image as a .jpeg file
            rgb_img.save(jpeg_path, "JPEG")

print("Conversion complete!")
