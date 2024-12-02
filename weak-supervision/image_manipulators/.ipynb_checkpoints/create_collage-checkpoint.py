from PIL import Image
import glob
from tqdm import tqdm
from collections import defaultdict
import os

def group_images_by_ending_number(folder_path):

    grouped_images = defaultdict(list)
    

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".tif"):
            name_without_ext = filename.split(".")[0]
            parts = name_without_ext.split("_")
            ending_number = parts[-1]
            
            middle_number = int(parts[1]) 
            full_path = os.path.join(folder_path, filename)
            
            grouped_images[ending_number].append((middle_number, full_path))
            
    for ending_number in grouped_images:
        grouped_images[ending_number].sort(key=lambda x: x[0])
        grouped_images[ending_number] = [filename for _, filename in grouped_images[ending_number]]
    
    return grouped_images

# def create_collage(images, grid_rows, grid_cols, padding=10):

#     img = Image.open(images[0])
#     img_width, img_height = img.size

#     collage_width = grid_cols * img_width + (grid_cols + 1) * padding
#     collage_height = grid_rows * img_height + (grid_rows + 1) * padding

#     collage = Image.new('RGB', (collage_width, collage_height), 'white')

#     for i, img_path in enumerate(images):
#         img = Image.open(img_path).resize((img_width, img_height), Image.LANCZOS)
#         x = (i % grid_cols) * (img_width + padding) + padding
#         y = (i // grid_cols) * (img_height + padding) + padding
#         collage.paste(img, (x,y))

#     return collage

def create_collage(images, grid_rows, grid_cols, image_path, padding=10):
    """
    Create a collage of images and save it to the specified folder with a given name.
    
    Parameters:
    - images (list): List of image paths.
    - grid_rows (int): Number of rows in the collage.
    - grid_cols (int): Number of columns in the collage.
    - output_folder (str): Destination folder to save the collage.
    - output_name (str): File name for the saved collage (without extension).
    - padding (int): Padding between images in the collage (default is 10).
    """
    # Ensure the output folder exists
    # os.makedirs(output_folder, exist_ok=True)
    
    # Open the first image to get dimensions
    img = Image.open(images[0])
    img_width, img_height = img.size

    # Calculate collage dimensions
    collage_width = grid_cols * img_width + (grid_cols + 1) * padding
    collage_height = grid_rows * img_height + (grid_rows + 1) * padding

    # Create a blank collage canvas
    collage = Image.new('RGB', (collage_width, collage_height), 'white')

    # Place each image in the grid
    for i, img_path in enumerate(images):
        img = Image.open(img_path).resize((img_width, img_height), Image.LANCZOS)
        x = (i % grid_cols) * (img_width + padding) + padding
        y = (i // grid_cols) * (img_height + padding) + padding
        collage.paste(img, (x, y))

    # Save the collage with the proper name in the output folder
    # output_path = os.path.join(image_path)
    collage.save(image_path)
    print(f"Collage saved to {image_path}")
    


folder_path = "/home/macula/SMATousi/Gullies/ground_truth/organized_data/All_Pos_Neg/combined_folder_true_rgb"
folder_to_save='/home/macula/SMATousi/Gullies/ground_truth/organized_data/All_Pos_Neg/combined_rgb_folder_collage_true'
os.makedirs(folder_to_save, exist_ok=True)

grouped_images = group_images_by_ending_number(folder_path)

for key in tqdm(grouped_images):
    # print(key)
    image_new_path=os.path.join(folder_to_save,f"{grouped_images[key][0].split('_')[-1].split('.')[0]}_collage_{key}.jpg")
    collage=create_collage(grouped_images[key], 2, 3, image_new_path, padding=10)
    
    # collage.save(image_new_path, "JPEG")
    # os.chmod(image_new_path, 0o666)
    # user_id = 569  # Replace with the target user's UID
    # group_id = 501  # Replace with the target group's GID
    # os.chown(image_new_path, user_id, group_id)


