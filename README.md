# Deep Learning Art School

Send your Pytorch model to art class!

This repository is both a framework and a set of tools for training deep neural networks that create images. It started as a branch of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk) but has been almost completely re-written at every level.

## Why do we need another training framework

These are a dime a dozen, no doubt. DL Art School (*DLAS*) differentiates itself by being configuration driven. You write the model code (specifically, a torch.nn.Module) and (possibly) some losses, then you cobble together a config file written in yaml that tells DLAS how to train it. Swapping model architectures is simple and often requires no changes to actual code. This effectively enables you to run multiple concurrent experiments that use the same codebase, as well as retain backwards compatibility for past experiments.

Training effective generators often means juggling multiple loss functions. As a result, DLAS' configuraion language is specifically designed to make it easy to support large number of losses and networks that interact with each other. As an example: some GANs I have trained in this framework consist of more than 15 losses and use 2 separate discriminators and require no bespoke code.

Generators are also notorious GPU memory hogs. I have spent substantial time streamlining the training framework to support gradient checkpointing and FP16. DLAS also supports "mega batching", where multiple forward passes contribute to a single backward pass. Most models can be trained on midrange GPUs with 8-11GB of memory.

The final value-added feature is interpretability. Tensorboard logging operates out of the box with no custom code. Intermediate images from within the training pipeline can be intermittently surfaced as normal PNG files so you can see what your network is up to. Validation passes are also cached as images so you can view how your network improves over time.

## Modeling Capabilities

DLAS was built with extensibly in mind. One of the reasons I'm putting in the effort to better document this code is the incredible ease with which I have been able to train entirely new model types with no changes to the core training code.

I intend to fill out the sections below with sample configurations which can be used to train different architectures. You will need to bring your own data.

### Super-resolution
TBC..
-  Pixel-based SR (SRCNN, RCAN, PANet, etc)
-  GAN-based SR (ESRGAN)
-  Multi-GAN SR (SPSR)
-  Video SR (TecoGAN)

### Optical Flow & 3D

### Style Transfer

## Dependencies and Installation

- Python 3
- [PyTorch >= 1.6](https://pytorch.org)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install -r requirements.txt`
- Some video utilities require [FFMPEG](https://ffmpeg.org/)

## User Guide
TBC

### Dataset Preparation
DLAS comes with some Dataset instances that I have created for my own use. Unless you want to use one of the recipes above, you'll need to provide your own. Here is how to add your own Dataset:

1.  Create a Dataset in codes/data/ which takes a single Python dict as a constructor and extracts options from that dict.
2.  Register your Dataset in codes/data/__init__.py
3.  Your Dataset should return a dict of tensors. The keys of the dict are injected directly into the training state, which you can interact within your configuration file.

### Training and Testing
There are currently 3 base scripts for interacting with models. They all take a single parameter, `-opt` which specifies the configuration file which controls how they work. Configs (will be) documented above in the user guide.

#### train.py
Starts (or continues) a training session.
`python train.py -opt <your_config.yml>`

#### test.py
Runs a model against a validation or test set of data and reports metrics (for now, just PSNR and a custom perceptual metric)
`python test.py -opt <your_config.yml>`

#### process_video.py
Breaks a video into individual frames and uses a network to do processing on it, then reassembles the output back into video form.
`python process_video -opt <your_config.yml>`

## Contributing
At this time I am not taking feature requests or bug reports, but I appreciate all contributions.

## License
This project is released under the Apache 2.0 license.
