# Working with Gaussian Diffusion models in DLAS

Diffusion Models are a method of generating structural data using a gradual de-noising process. This process allows a
simple network training regime.

This implementation of Gaussian Diffusion is largely based on the work done by OpenAI in their paper ["Diffusion Models
Beat GANs on Image Synthesis"](https://arxiv.org/pdf/2105.05233.pdf) and ["Improved Denoising Diffusion Probabilistic
Models"](https://arxiv.org/pdf/2102.09672).

OpenAI opened sourced their reference implementations [here](https://github.com/openai/guided-diffusion). The diffusion
model that DLAS trains uses the [gaussian_diffusion.py](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)
script from that repo for training and inference with these models. We also include the UNet from that repo as a model
that can be used to train a diffusion network.

Diffusion networks can be re-purposed to pretty much any image generation task, including super-resolution. Even though
they are trained with MSE losses, they produce incredibly crisp images with FID scores competitive with the best GANs.
More importantly, it is easy to track training progress since diffusion networks use a "normal" loss.

Diffusion networks are unique in that during inference, they perform multiple forward passes to generate a single image.
During training, these networks are trained to denoise images over 4000 steps. In inference, this sample rate can be
adjusted. For the purposes of super-resolution, I have found that images sampled in 50 steps to be of very good quality.
This still means that a diffusion generator is 50x slower than generators trained in different ways.

What's more is that I have found that diffusion networks can be trained in the tiled methodology used by ESRGAN: instead
of training on whole images, you can train on tiles of larger images. At inference time, the network can be applied to
larger images than the network was initially trained on. I have found this works well on inference images within ~3x
the training size. I have not tried larger, because the size of the UNet model means that inference at ultra-high 
resolutions is impossible (I run out of GPU memory).

I have provided a reference configuration for training a diffusion model in this manner. The config performs a 2x
upsampling to 256px, de-blurs it and removes JPEG artifacts. The deblurring and image repairs are done on a configurable
scale. The scale is [0,1] passed to the model as `corruption_entropy`. `1` represents a maximum correction factor.
You can try reducing this to 128px for faster training. It should work fine.

Diffusion models also have a fairly arcane inference method. To help you along, I've provided an inference configuration
that can be used with models trained in DLAS.