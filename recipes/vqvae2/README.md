# VQVAE2 in Pytorch

[VQVAE2](https://arxiv.org/pdf/1906.00446.pdf) is a generative autoencoder developed by Deepmind. It's unique innovation is
discretizing the latent space into a fixed set of "codebook" vectors.  This codebook
can then be used in downstream tasks to rebuild images from the training set.

This model is in DLAS thanks to work [@rosinality](https://github.com/rosinality) did 
[converting the Deepmind model](https://github.com/rosinality/vq-vae-2-pytorch) to Pytorch.

# Training VQVAE2

VQVAE2 is trained in two steps:

## Training the autoencoder

This first step is to train the autoencoder itself. The config file `train_imgnet_vqvae_stage1.yml` provided shows how to do this
for imagenet with the hyperparameters specified by deepmind. You'll need to bring your own imagenet folder for this.

## Training the PixelCNN encoder

The second step is to train the PixelCNN model which will create "codebook" vectors given an
input image.