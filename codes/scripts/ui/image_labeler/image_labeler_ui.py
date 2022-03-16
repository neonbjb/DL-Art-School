# Script that builds and launches a tkinter UI for directly interacting with a pytorch model that performs
# patch-level image classification.
#
# This script is a bit rough around the edges. I threw it together quickly to ascertain its usefulness. If I end up
# using it a lot, I may re-visit it.

import os
import tkinter as tk

import torch
import torchvision
from PIL import ImageTk

from data.images.image_label_parser import VsNetImageLabeler
from scripts.ui.image_labeler.pretrained_image_patch_classifier import PretrainedImagePatchClassifier

# Globals used to define state that event handlers might operate on.
classifier = None
gen = None
labeler = None
to_pil = torchvision.transforms.ToPILImage()
widgets = None
batch_gen = None
cur_img = 0
batch_sz = 0
mode = 0
cur_path, cur_top, cur_left, cur_dim = None, None, None, None
pending_labels = []


def update_mode_label():
    global widgets
    image_widget, primary_label, secondary_label, mode_label = widgets
    mode_label.config(text="Current mode: %s; Saved images: %i" % (labeler.str_labels[mode], len(pending_labels)))


# Handles the "change mode" hotkey. Changes the classification label being targeted.
def change_mode(event):
    global mode, pending_labels
    mode = (mode + 1) % len(labeler.str_labels)
    update_mode_label()


# Handles key presses, which are interpreted as requests to categorize a currently active image patch.
def key_press(event):
    global batch_gen, labeler, pending_labels

    if event.char != '\t':
        if event.char not in labeler.categories.keys():
            print("Specified category doesn't exist.")
            return
        cat = labeler.categories[event.char]
        img_name = os.path.basename(cur_path)
        print(cat['label'], img_name, cur_top, cur_left, cur_dim)
        pending_labels.append([event.char, img_name, cur_top, cur_left, cur_dim])

    if not next(batch_gen):
        next_image(None)
    update_mode_label()


# Pop the most recent label off of the stack.
def undo(event):
    global pending_labels
    c, nm, t, l, d = pending_labels.pop()
    print("Removed pending label", c, nm, t, l, d)
    update_mode_label()


# Save the stack of pending labels to the underlying label file.
def save(event):
    global pending_labels, labeler
    print("Saving %i labels", len(pending_labels,))
    for l in pending_labels:
        labeler.add_label(*l)
    labeler.save()
    pending_labels = []
    update_mode_label()


# This is the main controller for the state machine that this UI attaches to. At its core, it performs inference on
# a batch of images, then repetitively yields patches of images that fit within a confidence bound for the currently
# active label.
def next_batch():
    global gen, widgets, cur_img, batch_sz, cur_top, cur_left, cur_dim, mode, labeler, cur_path
    image_widget, primary_label, secondary_label, mode_label = widgets
    hq, res, data = next(gen)
    scale = hq.shape[-1] // res.shape[-1]

    # These are the confidence bounds. They apply to a post-softmax output. They are currently fixed.
    conf_lower = .4
    conf_upper = 1
    valid_res = ((res > conf_lower) * (res < conf_upper)) * 1.0  # This results in a tensor of 0's where tensors are outside of the confidence bound, 1's where inside.

    cur_img = 0
    batch_sz = hq.shape[0]
    while cur_img < batch_sz:  # Note: cur_img can (intentionally) be changed outside of this loop.
        # Build a random permutation for every image patch in the image. We will search for patches that fall within the confidence bound and yield them.
        permutation = torch.randperm(res.shape[-1] * res.shape[-2])
        for p in permutation:
            p = p.item()
        #for p in range(res.shape[-1]*res.shape[-2]):
            # Reconstruct a top & left coordinate.
            t = p // res.shape[-1]
            l = p % res.shape[-1]
            if not valid_res[cur_img,mode,t,l]:
                continue

            # Build a mask that shows the user the underlying construction of the image.
            # - Areas that don't fit the current labeling restrictions get a mask value of .25.
            # - Areas that do fit but are not the current patch, get a mask value of .5
            # - The current patch gets a mask value of 1.0
            # Expected output shape is (1,h,w) so it can be multiplied into the output image.
            mask = (valid_res[cur_img,mode,:,:].clone()*.25 + .25).unsqueeze(0)
            mask[:,t,l] = 1.0

            # Interpolate the mask so that it can be directly multiplied against the HQ image.
            masked = hq[cur_img,:,:,:].clone() * torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)

            # Update the image widget to show the new masked image.
            tk_picture = ImageTk.PhotoImage(to_pil(masked))
            image_widget.image = tk_picture
            image_widget.configure(image=tk_picture)

            # Fill in the labels
            probs = res[cur_img, :, t, l]
            probs, lblis = torch.topk(probs, k=2)
            primary_label.config(text="%s (p=%f)" % (labeler.str_labels[lblis[0].item()], probs[0]))
            secondary_label.config(text="%s (p=%f)" % (labeler.str_labels[lblis[1].item()], probs[1]))

            # Update state variables so that the key handlers can save the current patch as needed.
            cur_top, cur_left, cur_dim = (t*scale), (l*scale), scale
            cur_path = os.path.basename(data['HQ_path'][cur_img])
            yield True
        cur_img += 1
    cur_top, cur_left, cur_dim = None, None, None
    return False


def next_image(event):
    global batch_gen, batch_sz, cur_img
    cur_img += 1
    if cur_img >= batch_sz:
        cur_img = 0
        batch_gen = next_batch()
    next(batch_gen)


if __name__ == '__main__':
    classifier = PretrainedImagePatchClassifier('../options/train_imgset_structural_classifier.yml')
    gen = classifier.get_next_sample()
    labeler = VsNetImageLabeler('F:\\4k6k\\datasets\\ns_images\\512_unsupervised\\categories_new.json')

    window = tk.Tk()
    window.title("Image labeler UI")
    window.geometry('512x620+100+100')

    # Photo view.
    image_widget = tk.Label(window)
    image_widget.place(x=0, y=0, width=512, height=512)

    # Labels
    primary_label = tk.Label(window, text="xxxx (p=1.0)", anchor="w")
    primary_label.place(x=20, y=510, width=400, height=20)
    secondary_label = tk.Label(window, text="yyyy (p=0.0)", anchor="w")
    secondary_label.place(x=20, y=530, width=400, height=20)
    help = tk.Label(window, text="Next: ctrl+f, Mode: ctrl+x, Undo: ctrl+z, Save: ctrl+s", anchor="w")
    help.place(x=20, y=550, width=400, height=20)
    help2 = tk.Label(window, text=','.join(list(labeler.categories.keys())), anchor="w")
    help2.place(x=20, y=570, width=400, height=20)
    mode_label = tk.Label(window, text="Current mode: %s; Saved images: %i" % (labeler.str_labels[mode], 0), anchor="w")
    mode_label.place(x=20, y=590, width=400, height=20)

    widgets = (image_widget, primary_label, secondary_label, mode_label)

    window.bind("<Control-x>", change_mode)
    window.bind("<Control-z>", undo)
    window.bind("<Control-s>", save)
    window.bind("<Control-f>", next_image)
    for kb in labeler.categories.keys():
        window.bind("%s" % (kb,), key_press)
    window.bind("<Tab>", key_press)  # Skip current patch
    window.mainloop()
