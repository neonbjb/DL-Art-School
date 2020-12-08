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

Note: My intent is to adapt BYOL for use on structured models - e.g. models that do *not* collapse
the latent into a flat map. Stay tuned for that..

# Training BYOL

In this directory, you will find a sample training config for training BYOL on DIV2K. You will
likely want to insert your own model architecture first. Exchange out spinenet for your
model architecture and change the `hidden_layer` parameter to a layer from your network
that you want the BYOL model wrapper to hook into. 

*hint: Your network architecture (including layer names) is printed out when running train.py
against your network.*

Run the trainer by:

`python train.py -opt train_div2k_byol.yml`