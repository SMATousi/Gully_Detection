import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
np.random.seed(0)
# Define the set of rgb numbers (8 images per tile)
RGB_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 11]

class LabelingGUI:
    def __init__(self, master, labeler_name, assigned_tiles):
        self.master = master
        self.labeler_name = labeler_name
        self.assigned_tiles = assigned_tiles  # Tiles assigned to the current labeler
        self.master.title(f"Ephemeral Gully Presence Labeling - {self.labeler_name}")
        # JSON file for this labeler
        self.json_filename = f"{self.labeler_name}.json"
        self.data = {}
        self.load_json()

        # Tile numbers assigned to the labeler
        self.tile_numbers = assigned_tiles
        self.current_tile = None
        self.photo_images = {}  # to keep references to image objects

        # Set up two main areas: a left panel for tile navigation and a right panel for labeling.
        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_tile_list()
        self.create_labeling_panel()
        self.create_navigation_buttons()
        self.create_progress_bar()

    def load_json(self):
        if os.path.exists(self.json_filename):
            try:
                with open(self.json_filename, "r") as f:
                    self.data = json.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON file: {e}")
                self.data = {}
        else:
            self.data = {}

    def save_json(self):
        try:
            with open(self.json_filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON file: {e}")

    def create_tile_list(self):
        tk.Label(self.left_frame, text="Tile Numbers").pack()
        self.tile_listbox = tk.Listbox(self.left_frame, width=15, height=25)
        self.tile_listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.tile_listbox.bind("<<ListboxSelect>>", self.on_tile_select)
        self.update_tile_listbox()

    def update_tile_listbox(self):
        self.tile_listbox.delete(0, tk.END)
        for tile in self.tile_numbers:
            display_text = str(tile)
            if str(tile) in self.data:
                display_text += "  ✔"
            self.tile_listbox.insert(tk.END, display_text)

    def create_labeling_panel(self):
        # Frame for displaying images
        self.image_frame = tk.Frame(self.right_frame)
        self.image_frame.pack(pady=10)

        # Overall label frame with radio buttons (0 to 4)
        self.overall_label_frame = tk.Frame(self.right_frame)
        self.overall_label_frame.pack(pady=10)
        tk.Label(self.overall_label_frame, text="Overall Label (0: no EG, 4: EG)").pack(side=tk.LEFT)

        self.overall_label_var = tk.IntVar()
        for i in range(5):
            rb = tk.Radiobutton(self.overall_label_frame, text=str(i),
                                variable=self.overall_label_var, value=i,
                                command=self.on_overall_label_change)
            rb.pack(side=tk.LEFT, padx=5)

        # Frame for per-image checkboxes (only active if overall label is 3 or 4)
        self.checkbox_frame = tk.Frame(self.right_frame)
        self.checkbox_frame.pack(pady=10)

        self.checkbox_vars = {}
        self.checkbox_widgets = {}
        self.image_labels = {}
        # Arrange images in a 2-row x 4-column grid
        for idx, rgb in enumerate(RGB_NUMBERS):
            row = idx // 4
            col = idx % 4
            frame = tk.Frame(self.image_frame, borderwidth=2, relief=tk.RIDGE)
            frame.grid(row=row, column=col, padx=5, pady=5)

            img_label = tk.Label(frame)
            img_label.pack()
            self.image_labels[rgb] = img_label

            var = tk.BooleanVar()
            cb = tk.Checkbutton(frame, text="Choose", variable=var)
            cb.pack()
            cb.config(state=tk.DISABLED)
            self.checkbox_vars[rgb] = var
            self.checkbox_widgets[rgb] = cb

    def create_navigation_buttons(self):
        nav_frame = tk.Frame(self.right_frame)
        nav_frame.pack(pady=10)
        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.goto_previous_tile)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = tk.Button(nav_frame, text="Next", command=self.goto_next_tile)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.save_next_button = tk.Button(nav_frame, text="Save & Next", command=self.save_and_next)
        self.save_next_button.pack(side=tk.LEFT, padx=5)

    def create_progress_bar(self):
        # Progress bar to track the labeling progress
        self.progress_bar_label = tk.Label(self.right_frame, text="Progress:")
        self.progress_bar_label.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.right_frame, length=300, mode="determinate", maximum=len(self.tile_numbers))
        self.progress_bar.pack(pady=5)

    def update_progress_bar(self):
        labeled_tiles = len([tile for tile in self.tile_numbers if str(tile) in self.data])
        self.progress_bar["value"] = labeled_tiles
        self.master.update_idletasks()

    def on_tile_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            tile_str = self.tile_listbox.get(index).split()[0]  # remove check mark if present
            self.current_tile = tile_str
            self.load_tile(tile_str)

    def load_tile(self, tile_number):
        # Load and display images for the selected tile
        root_path = './data/all_tiles_25/'
        for rgb in RGB_NUMBERS:
            filename = f"rgb_{rgb}_tile_{tile_number}.tif"
            filepath = os.path.join(root_path,filename)
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath)
                    img = img.resize((350, 350))
                    img.thumbnail((500, 500))
                    
                    photo = ImageTk.PhotoImage(img)
                    self.photo_images[rgb] = photo  # keep reference
                    self.image_labels[rgb].config(image=photo, text="")
                except Exception as e:
                    self.image_labels[rgb].config(text="Error loading image", image="")
            else:
                self.image_labels[rgb].config(text="Image not found", image="")

        # Load any saved labels for this tile, if available.
        tile_data = self.data.get(tile_number, {})
        overall = tile_data.get("overall", 0)
        self.overall_label_var.set(overall)
        image_labels_saved = tile_data.get("images", {})
        for rgb in RGB_NUMBERS:
            val = image_labels_saved.get(str(rgb), False)
            self.checkbox_vars[rgb].set(val)

        self.on_overall_label_change()

    def on_overall_label_change(self):
        # Enable checkboxes only if overall label is 3 or 4.
        val = self.overall_label_var.get()
        state = tk.NORMAL if val in (3, 4) else tk.DISABLED
        if state == tk.DISABLED:
            for rgb in RGB_NUMBERS:
                self.checkbox_vars[rgb].set(False)
        for rgb in RGB_NUMBERS:
            self.checkbox_widgets[rgb].config(state=state)

    def save_current_label(self):
        if self.current_tile is None:
            messagebox.showwarning("No tile selected", "Please select a tile first.")
            return

        overall = self.overall_label_var.get()
        image_labels = {}
        if overall in (3, 4):
            for rgb in RGB_NUMBERS:
                image_labels[str(rgb)] = self.checkbox_vars[rgb].get()

        self.data[self.current_tile] = {"overall": overall, "images": image_labels}
        self.save_json()
        messagebox.showinfo("Saved", f"Tile {self.current_tile} saved.")
        self.update_tile_listbox()
        self.update_progress_bar()

    def save_and_next(self):
        self.save_current_label()
        self.goto_next_tile()

    def goto_next_tile(self):
        if self.current_tile is None:
            # If nothing is selected, start at the first tile.
            self.current_tile = str(self.tile_numbers[0])
            self.load_tile(self.current_tile)
            self.tile_listbox.selection_clear(0, tk.END)
            self.tile_listbox.selection_set(0)
            self.tile_listbox.activate(0)
            return

        try:
            current_index = self.tile_numbers.index(int(self.current_tile))
        except ValueError:
            current_index = 0

        if current_index < len(self.tile_numbers) - 1:
            next_tile = self.tile_numbers[current_index + 1]
            self.current_tile = str(next_tile)
            self.load_tile(self.current_tile)
            self.tile_listbox.selection_clear(0, tk.END)
            self.tile_listbox.selection_set(current_index + 1)
            self.tile_listbox.activate(current_index + 1)
        else:
            messagebox.showinfo("Info", "This is the last tile.")

    def goto_previous_tile(self):
        if self.current_tile is None:
            return

        try:
            current_index = self.tile_numbers.index(int(self.current_tile))
        except ValueError:
            current_index = 0

        if current_index > 0:
            prev_tile = self.tile_numbers[current_index - 1]
            self.current_tile = str(prev_tile)
            self.load_tile(self.current_tile)
            self.tile_listbox.selection_clear(0, tk.END)
            self.tile_listbox.selection_set(current_index - 1)
            self.tile_listbox.activate(current_index - 1)
        else:
            messagebox.showinfo("Info", "This is the first tile.")

def choose_labeler(master):
    # Allow the user to choose their name from a predefined list.
    login_frame = tk.Frame(master)
    login_frame.pack(padx=20, pady=20)
    tk.Label(login_frame, text="Select your labeler:").pack(pady=5)

    labeler_names = ["Ali", "Dr.Lory", "Krystal"]
    selected_labeler = tk.StringVar(value=labeler_names[0])
    combobox = ttk.Combobox(login_frame, textvariable=selected_labeler,
                            values=labeler_names, state="readonly", width=20)
    combobox.pack(pady=5)

    def on_ok():
        login_frame.destroy()  # Remove the login frame and proceed
    tk.Button(login_frame, text="OK", command=on_ok).pack(pady=5)

    master.wait_window(login_frame)  # Wait until the login frame is closed
    return selected_labeler.get()

def get_assigned_tiles(labeler_name):
    # Define a set of tiles assigned to each labeler.
    total_tiles = list(range(0,500))
    np.random.shuffle(total_tiles)
    # labele_count = 3

    assignments = {
        "Ali": total_tiles[:150],
        "Dr.Lory": total_tiles[150:300],
        "Krystal": total_tiles[300:]
    }
    return assignments.get(labeler_name, [])

def main():
    root = tk.Tk()
    labeler_name = choose_labeler(root)
    assigned_tiles = get_assigned_tiles(labeler_name)
    app = LabelingGUI(root, labeler_name, assigned_tiles)
    root.mainloop()

if __name__ == "__main__":
    main()
