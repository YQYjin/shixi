import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk



class ImageComparer:
    def __init__(self, root, left_folder, right_folder):
        self.root = root
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.left_images = sorted(os.listdir(left_folder), key=lambda x: int(os.path.splitext(x)[0]))
        self.right_images = sorted(os.listdir(right_folder), key=lambda x: int(os.path.splitext(x)[0]))
        self.left_index = 0
        self.right_index = 0

        self.init_gui()

    def init_gui(self):
        self.root.title("Image Comparer")

        # Frame for left image
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.left_image_label = ttk.Label(self.left_frame)
        self.left_image_label.pack()

        # Frame for right image
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)
        self.right_image_label = ttk.Label(self.right_frame)
        self.right_image_label.pack()

        # Controls frame
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.next_button = ttk.Button(self.controls_frame, text="下一个", command=self.next_image)
        self.next_button.grid(row=0, column=0, padx=5)

        self.row_label = ttk.Label(self.controls_frame, text="行号")
        self.row_label.grid(row=0, column=1, padx=5)
        self.row_entry = ttk.Entry(self.controls_frame)
        self.row_entry.grid(row=0, column=2, padx=5)

        self.column_label = ttk.Label(self.controls_frame, text="列号")
        self.column_label.grid(row=0, column=3, padx=5)
        self.column_entry = ttk.Entry(self.controls_frame)
        self.column_entry.grid(row=0, column=4, padx=5)

        self.correct_button = ttk.Button(self.controls_frame, text="正确", command=self.mark_correct)
        self.correct_button.grid(row=0, column=5, padx=5)

        self.incorrect_button = ttk.Button(self.controls_frame, text="错误", command=self.mark_incorrect)
        self.incorrect_button.grid(row=0, column=6, padx=5)

        # Display the first images
        self.display_images()

    def display_images(self):
        left_image_path = os.path.join(self.left_folder, self.left_images[self.left_index])
        right_image_path = os.path.join(self.right_folder, self.right_images[self.right_index])

        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)

        left_image.thumbnail((400, 400))
        right_image.thumbnail((400, 400))

        left_photo = ImageTk.PhotoImage(left_image)
        right_photo = ImageTk.PhotoImage(right_image)

        self.left_image_label.config(image=left_photo)
        self.left_image_label.image = left_photo

        self.right_image_label.config(image=right_photo)
        self.right_image_label.image = right_photo

    def next_image(self):
        self.left_index = (self.left_index + 1) % len(self.left_images)
        self.right_index = (self.right_index + 1) % len(self.right_images)
        self.display_images()

    def mark_correct(self):
        row = self.row_entry.get()
        column = self.column_entry.get()
        if row and column:
            new_name = f"0_{row}_{column}.png"
            self.rename_image(new_name)

    def mark_incorrect(self):
        row = self.row_entry.get()
        column = self.column_entry.get()
        if row and column:
            new_name = f"1_{row}_{column}.png"
            self.rename_image(new_name)

    def rename_image(self, new_name):
        current_image_path = os.path.join(self.right_folder, self.right_images[self.right_index])
        new_image_path = os.path.join(self.right_folder, new_name)
        os.rename(current_image_path, new_image_path)
        self.right_images[self.right_index] = new_name
        self.display_images()

def main():
    root = tk.Tk()
    left_folder = filedialog.askdirectory(title="选择左侧图片文件夹")
    right_folder = filedialog.askdirectory(title="选择右侧图片文件夹")
    app = ImageComparer(root, left_folder, right_folder)
    root.mainloop()

if __name__ == "__main__":
    main()
