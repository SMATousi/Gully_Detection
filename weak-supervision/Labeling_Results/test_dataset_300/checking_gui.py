import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import glob

# Set the path to the directory containing the images
image_dir = "/home/macula/SMATousi/cluster/docker-images/ollama/collage"

# Load tile data from the JSON file
with open('./GT_L6_Tol.json', 'r') as file:
    tile_data = json.load(file)


class TileLabelingApp:
    def __init__(self, root, tile_data):
        self.root = root
        self.root.title("Tile Labeling Agreement Tool")
        self.tile_data = tile_data
        self.tiles = list(tile_data.keys())
        self.total_tiles = len(self.tiles)
        self.current_index = 0
        self.agree_count = 0
        self.disagree_count = 0

        # Setup UI components
        self.label_frame = tk.LabelFrame(root, text="Tile Information", padx=10, pady=10)
        self.label_frame.pack(padx=10, pady=10)
        
        self.image_label = tk.Label(self.label_frame)
        self.image_label.grid(row=0, column=0, columnspan=2)
        
        self.label_info = tk.Label(self.label_frame, text="")
        self.label_info.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        self.labelers_info = tk.Label(self.label_frame, text="")
        self.labelers_info.grid(row=2, column=0, columnspan=2, pady=(5, 10))
        
        # Progress label
        self.progress_label = tk.Label(root, text="")
        self.progress_label.pack(pady=(5, 10))

        # Buttons for Agree and Disagree
        self.agree_button = tk.Button(root, text="Agree", command=self.agree)
        self.agree_button.pack(side="left", padx=(10, 5), pady=10)
        
        self.disagree_button = tk.Button(root, text="Disagree", command=self.disagree)
        self.disagree_button.pack(side="right", padx=(5, 10), pady=10)

        self.update_tile()

    def update_tile(self):
        if self.current_index < self.total_tiles:
            tile_number = self.tiles[self.current_index]
            label = self.tile_data[tile_number]["label"]
            labelers = ", ".join(self.tile_data[tile_number]["labelers"])

            # Load and display the image with a flexible filename
            image_path = self.find_image(tile_number)
            if image_path:
                img = Image.open(image_path)
                img = img.resize((800, 800), Image.ANTIALIAS)  # Resize to fit the UI
                self.img_display = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.img_display)
                self.image_label.image = self.img_display
            else:
                self.image_label.config(text="Image not found", image="")

            # Update label and progress information
            self.label_info.config(text=f"Tile Number: {tile_number}, Label: {label}")
            self.labelers_info.config(text=f"Labelers: {labelers}")
            self.progress_label.config(text=f"Progress: {self.current_index + 1}/{self.total_tiles}")
        else:
            self.show_results()

    def find_image(self, tile_number):
        # Look for a file with the pattern `*_collage_{tile_number}.jpg` in the images directory
        pattern = os.path.join(image_dir, f"*collage_{tile_number}.jpg")
        matching_files = glob.glob(pattern)
        return matching_files[0] if matching_files else None

    def agree(self):
        self.agree_count += 1
        self.next_tile()

    def disagree(self):
        self.disagree_count += 1
        self.next_tile()

    def next_tile(self):
        self.current_index += 1
        self.update_tile()

    def show_results(self):
        total = self.agree_count + self.disagree_count
        agree_percentage = (self.agree_count / total) * 100 if total > 0 else 0
        disagree_percentage = (self.disagree_count / total) * 100 if total > 0 else 0
        messagebox.showinfo("Results", f"Agree: {agree_percentage:.2f}%\nDisagree: {disagree_percentage:.2f}%")
        self.root.quit()

# Main application execution
root = tk.Tk()
app = TileLabelingApp(root, tile_data)
root.mainloop()