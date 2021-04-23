# Script that builds and launches a tkinter UI for labeling similar points between two images.
import os
import tkinter as tk
from glob import glob
from random import choices

import torch
from PIL import ImageTk, Image

# Globals used to define state that event handlers might operate on.
imgs_list = []
widgets = None
cur_img_1, cur_img_2 = None, None
pil_img_1, pil_img_2 = None, None
pending_labels = []
mode_select_image_1 = True
img_count = 1
img_loc_1 = None
output_location = "results"


def update_mode_label():
    global widgets, mode_select_image_1, img_count
    image_widget_1, image_widget_2, mode_label = widgets
    mode_str = "Select point in image 1" if mode_select_image_1 else "Select point in image 2"
    mode_label.config(text="%s; Saved images: %i" % (mode_str, img_count))


# Handles key presses, which are interpreted as requests to categorize a currently active image patch.
def key_press(event):
    global batch_gen, labeler, pending_labels

    if event.char == '\t':
        next_images()

    update_mode_label()


def click(event):
    global img_loc_1, mode_select_image_1, pil_img_1, pil_img_2, img_count
    x, y = event.x, event.y
    if x > 512 or y > 512:
        print(f"Bounds error {x} {y}")
        return

    print(f"Detected click. {x} {y}")
    if mode_select_image_1:
        img_loc_1 = x, y
        mode_select_image_1 = False
    else:
        ofolder = f'{output_location}/{img_count}'
        os.makedirs(ofolder)
        pil_img_1.save(os.path.join(ofolder, "1.jpg"))
        pil_img_2.save(os.path.join(ofolder, "2.jpg"))
        torch.save([img_loc_1, (x,y)], os.path.join(ofolder, "coords.pth"))
        img_count = img_count + 1
        mode_select_image_1 = True
        next_images()
    update_mode_label()


def load_image_into_pane(img_path, pane, size=512):
    pil_img = Image.open(img_path)
    pil_img = pil_img.resize((size,size))
    tk_picture = ImageTk.PhotoImage(pil_img)
    pane.image = tk_picture
    pane.configure(image=tk_picture)
    return pil_img

def next_images():
    global imgs_list, widgets, cur_img_1, cur_img_2, pil_img_1, pil_img_2
    image_widget_1, image_widget_2, mode_label = widgets

    cur_img_1, cur_img_2 = choices(imgs_list, k=2)
    pil_img_1 = load_image_into_pane(cur_img_1, image_widget_1)
    pil_img_2 = load_image_into_pane(cur_img_2, image_widget_2)

if __name__ == '__main__':
    os.makedirs(output_location, exist_ok=True)

    window = tk.Tk()
    window.title("Image pair labeler UI")
    window.geometry('1024x620+100+100')

    # Load images
    imgs_list = glob("E:\\4k6k\\datasets\\ns_images\\imagesets\\imageset_1024_square_with_new\\*.jpg")

    # Photo view.
    image_widget_1 = tk.Label(window)
    image_widget_1.place(x=0, y=0, width=512, height=512)
    image_widget_2 = tk.Label(window)
    image_widget_2.place(x=512, y=0, width=512, height=512)

    # Labels
    mode_label = tk.Label(window, text="", anchor="w")
    mode_label.place(x=20, y=590, width=400, height=20)

    widgets = (image_widget_1, image_widget_2, mode_label)

    window.bind("<Tab>", key_press)  # Skip current patch
    window.bind("<Button-1>", click)
    next_images()
    update_mode_label()
    window.mainloop()
