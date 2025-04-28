import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Define the set of rgb numbers (8 images per tile)
RGB_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 11]

class ReviewToolGUI:
    def __init__(self, master, aggregated_file, image_folder="all_tiles_25"):
        self.master = master
        self.master.title("Review Aggregated Results")
        self.image_folder = image_folder

        # Load aggregated results from JSON file
        try:
            with open(aggregated_file, "r") as f:
                self.data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load aggregated JSON: {e}")
            self.data = {}

        # Get sorted list of tile numbers from the JSON keys (as strings)
        self.tile_numbers = sorted(self.data.keys(), key=lambda x: int(x))
        self.current_tile = None
        self.photo_images = {}  # to keep references to image objects

        # Layout: left panel for tile navigation, right panel for review display.
        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_tile_list()
        self.create_review_panel()

    def create_tile_list(self):
        tk.Label(self.left_frame, text="Tile Numbers").pack()
        self.tile_listbox = tk.Listbox(self.left_frame, width=15, height=25)
        self.tile_listbox.pack(side=tk.LEFT, fill=tk.Y)
        for tile in self.tile_numbers:
            self.tile_listbox.insert(tk.END, tile)
        self.tile_listbox.bind("<<ListboxSelect>>", self.on_tile_select)

    def create_review_panel(self):
        # This frame will be updated each time a tile is selected.
        self.review_container = tk.Frame(self.right_frame)
        self.review_container.pack(fill=tk.BOTH, expand=True)

    def clear_review_panel(self):
        # Remove all widgets in the review container
        for widget in self.review_container.winfo_children():
            widget.destroy()

    def on_tile_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            tile = self.tile_listbox.get(index)
            self.current_tile = tile
            self.show_tile_results(tile)

    def show_tile_results(self, tile):
        self.clear_review_panel()
        # Get the aggregated data for the tile
        tile_data = self.data.get(tile, {})
        images_data = tile_data.get("images", {})

        # We'll display a grid (2 rows x 4 columns) for the images in RGB_NUMBERS order.
        for idx, rgb in enumerate(RGB_NUMBERS):
            row = idx // 4
            col = idx % 4
            frame = tk.Frame(self.review_container, borderwidth=2, relief=tk.RIDGE)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            # Attempt to load the image file
            filename = f"rgb_{rgb}_tile_{tile}.tif"
            filepath = os.path.join(self.image_folder, filename)
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath)
                    # Resize image for display (adjust sizes as needed)
                    img = img.resize((250, 250), resample=Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.photo_images[(tile, rgb)] = photo  # keep reference
                    img_label = tk.Label(frame, image=photo)
                    img_label.pack()
                except Exception as e:
                    tk.Label(frame, text="Error loading image").pack()
            else:
                tk.Label(frame, text="Image not found").pack()

            # Retrieve aggregated labels for this image.
            # The aggregated JSON has keys as strings. Use str(rgb) for lookup.
            labels_dict = images_data.get(str(rgb), {})
            # Create a string summarizing the labels
            if labels_dict:
                summary = ", ".join(f"{name}: {val}" for name, val in labels_dict.items())
            else:
                summary = "No data"
            tk.Label(frame, text=summary, wraplength=250, justify=tk.LEFT).pack(pady=5)

def main():
    root = tk.Tk()
    # Path to your aggregated JSON file
    aggregated_file = "aggregated_results.json"
    app = ReviewToolGUI(root, aggregated_file)
    root.mainloop()

if __name__ == "__main__":
    main()
