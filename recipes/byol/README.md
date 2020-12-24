# Working with BYOL in DLAS

[BYOL](https://arxiv.org/abs/2006.07733) is a technique for pretraining an arbitrary image processing
neural network. It is built upon previous self-supervised architectures like SimCLR.

BYOL in DLAS is adapted from an implementation written by [lucidrains](https://github.com/lucidrains/byol-pytorch).
It is implemented via two wrappers: 

1. A Dataset wrapper that augments the LQ and HQ inputs from a typical DLAS dataset. Since differentiable
   augmentations don't actually matter for BYOL, it makes more sense (to me) to do this on the CPU at the
   dataset layer, so your GPU can focus on processing gradients.
1. A model wrapper that attaches a small MLP to the end of your input network to produce a fixed
   size latent. This latent is used to produce the BYOL loss which trains the master weights from
   your network.
   
Thanks to the excellent implementation from lucidrains, this wrapping process makes training your
network on unsupervised datasets extremely easy.

The DLAS version improves on lucidrains implementation adding some important training details, such as
a custom LARS optimizer implementation that aligns with the recommendations from the paper. By moving augmentation
to the dataset level, additional augmentation options are unlocked - like being able to take two similar video frames
as the image pair.

# Training BYOL

In this directory, you will find a sample training config for training BYOL on DIV2K. You will
likely want to insert your own model architecture first.

Run the trainer by:

`python train.py -opt train_div2k_byol.yml`

BYOL is data hungry, as most unsupervised training methods are. If you're providing your own dataset, make sure it is
the hundreds of K-images or more!

## Using your own model

Training your own model on this BYOL implementation is trivial:
1. Add your nn.Module model implementation to the models/ directory.
2. Register your model with `trainer/networks.py` as a generator. This file tells DLAS how to build your model from
   a set of configuration options.
3. Copy the sample training config. Change the `subnet` and `hidden_layer` params.
4. Run your config with `python train.py -opt <your_config>`.

*hint: Your network architecture (including layer names) is printed out when running train.py
against your network.*