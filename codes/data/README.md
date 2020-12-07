# DLAS Datasets

## Quick Overview

DLAS uses the standard Torch Dataset infrastructure. Datasets are expected to be constructed using an "options" dict,
which is fed directly from the configuration file. They are also expected to output a dict, where the keys are injected
directly into the trainer state.

Datasets conforming to the above expectations must be registered in `__init__.py` to be used by a configuration.

## Reference Datasets

This directory contains several reference datasets which I have used in building DLAS. They include:

1. Stylegan2Dataset - Reads a set of images from a directory, performs some basic augmentations on them and injects
   them directly into the state. LQ = HQ in this dataset.
1. SingleImageDataset - Reads image patches from a 'chunked' format along with the reference image and metadata about
   how the patch was originally computed. The 'chunked' format is described below. Includes built-in ImageCorruption
   features actuated by `image_corruptor.py`.
1. MultiframeDataset - Similar to SingleImageDataset, but infers a temporal relationship between images based on their
   filenames: the last 12 characters before the file extension are assumed to be a frame counter. Images from this 
   dataset are grouped together with a temporal dimension for working with video data.
1. ImageFolderDataset - Reads raw images from a folder and feeds them into the model. Capable of performing corruptions
   on those images like the above.
1. MultiscaleDataset - Reads full images from a directory and builds a tree of images constructed by cropping squares
   from the source image and resizing them to the target size recursively until the native resolution is hit. Each
   recursive step decreases the crop size by a factor of 2.
1. TorchDataset - A wrapper for miscellaneous pytorch datasets (e.g. MNIST, CIFAR, etc) which extracts the images
   and reformats them in a way that the DLAS trainer understands.
1. FullImageDataset - An image patch dataset where the patches are dynamically extracted from full-size images. I have
   generally stopped using this for performance reasons and it should be considered deprecated.
   
## Information about the "chunked" format

This is the main format I have used in my experiments with image super resolution. It is fast to read and provides
rich metadata on the images that the patches are derived from, including a downsized "reference" fullsize image and
information on where the crop was taken from in the original image.

### Creating a chunked dataset

The file format for 'chunked' datasets is very particular. I recommend using `scripts/extract_subimages_with_ref.py`
to build these datasets from raw images. Here is how you would do that:

1. Edit `scripts/extract_subimages_with_ref.py` to set these configuration options:
    ```
    opt['input_folder'] = <path to raw images>
    opt['save_folder'] = <where your chunked dataset will be stored>
    opt['crop_sz'] = [256, 512]  # A list, the size of each sub-image that will be extracted and turned into patches.
    opt['step'] = [128, 256]  # The pixel distance the algorithm will step for each sub-image. If this is < crop_sz, patches will share image content.
    opt['thres_sz'] = 128  # Amount of space that must be present on the edges of an image for it to be included in the image patch. Generally should be equal to the lowest step size.
    opt['resize_final_img'] = [1, .5] # Reduction factor that will be applied to image patches at this crop_sz level. TODO: infer this.
    opt['only_resize'] = False # If true, disables the patch-removal algorithm and just resizes the input images.
    opt['vertical_split'] = False # Used for stereoscopic images. Not documented.
    ```
   Note: the defaults should work fine for many applications.
1. Execute the script: `python scripts/extract_subimages_with_ref.py`. If you are having issues with imports, make sure
   you set `PYTHONPATH` to the repo root.

### Chunked cache

To make trainer startup fast, the chunked datasets perform some preprocessing the first time they are loaded. The entire
dataset is scanned and a cache is built up and saved in cache.pth. Future invocations only need to load cache.pth on
startup, which greatly speeds up trainer startup when you are debugging issues.

There is an important caveat here: this cache will not be recomputed unless you delete it. This means if you add new
images to your dataset, you must delete the cache for them to be picked up! Likewise, if you copy your dataset to a
new file path or a different computer, cache.pth must be deleted for it to work. In the latter case, you'll likely run 
into some weird errors.

### Details about the dataset format

If you look inside of a dataset folder output by above, you'll see a list of folders. Each folder represents a single
image that was found by the script.

Inside of that folder, you will see 3 different types of files:

1. Image patches, each of which have a unique ID within the given set. These IDs do not necessarily need to be unique
   across the entire dataset.
1. `centers.pt` A pytorch pickle which is just a dict that describes some metadata about the patches, like: where they
   were located in the source image and their original width/height.
1. `ref.jpg` Is a square version of the original image that is downsampled to the patch size.