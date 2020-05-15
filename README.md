# MMSR

MMSR is an open source image and video super-resolution toolbox based on PyTorch. It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk). MMSR is based on our previous projects: [BasicSR](https://github.com/xinntao/BasicSR), [ESRGAN](https://github.com/xinntao/ESRGAN), and [EDVR](https://github.com/xinntao/EDVR).

## My (@neonbjb) Modifications
After tinkering with MMSR, I really began to like a lot about how the codebase was laid out and the general practices being used. I have since worked to extend it to more
general use cases, as well as implement several GAN training features. The additions are too many to list, but I'll give it a shot:

- FP16 support.
- Alternative dataset support (notably a disjoint dataset for training a generator to style-transfer between imagesets).
- Addition of several new architectures, including a ResNet-based discrimator, a downsampling generator (for training image corruptors), and a fix-and-upsample generator.
- Fixup resblock support which resists the exploding gradients which necessitate batch norms. Most of the fixup architectures can be trained with BN turned off, though they
  take longer to train and are occasionally divergent in FP16 mode.
- Batch testing for performing generator augmentation on large sets of images.
- Model swapout during training - randomly select a past D or G and substitute it in for a short time to increase variance on the respective model.
- Adding random noise on both the inputs of the discriminator and generator. The discriminator variety has a decay.
- Decaying the influence of the feature loss.
- "Corruption" generators which can alter an input before it is fed through the SRGAN pipeline.
- Outputting "state" images which are very useful in debugging what is actually going on in the pipeline.
- Skip layers between the generator and discriminator.
- Support for any number of image resolutions into the discriminators. The original MMSR only accepted 128x128 images.
- "Megabatches" - gradient accumulation across multiple batches before performing an optimizer step.
- Image cropping can be disabled. I prefer to do this in preprocessing.
- Tensorboard logs for an experiment are cleared out when the experiment is restarted anew.
- A LOT more data is logged to tensorboard.

Note that this codebase is far from clean. I've notably broken LMDB support in a couple of places. Likely everything other than SRGAN doesn't work too well anymore either.
I will get around to documenting all this in the near future once the repo stabilizes a bit. For now, you're on your own!

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download))
- [PyTorch >= 1.1](https://pytorch.org)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [mmdetection](https://github.com/open-mmlab/mmdetection)'s dcn implementation. Please first compile it.
  ```
  cd ./codes/models/archs/dcn
  python setup.py develop
  ```
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Please refer to [DATASETS.md](datasets/DATASETS.md) for more details.

## Training and Testing
Please see [wiki- Training and Testing](https://github.com/open-mmlab/mmsr/wiki/Training-and-Testing) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [wiki-Model Zoo](https://github.com/open-mmlab/mmsr/wiki/Model-Zoo).

## Contributing
We appreciate all contributions. Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/CONTRIBUTING.md) for contributing guideline.

**Python code style**<br/>
We adopt [PEP8](https://python.org/dev/peps/pep-0008) as the preferred code style. We use [flake8](http://flake8.pycqa.org/en/latest) as the linter and [yapf](https://github.com/google/yapf) as the formatter. Please upgrade to the latest yapf (>=0.27.0) and refer to the [yapf configuration](.style.yapf) and [flake8 configuration](.flake8).

> Before you create a PR, make sure that your code lints and is formatted by yapf.

## License
This project is released under the Apache 2.0 license.
