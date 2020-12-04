# Working with SRFlow in DLAS

[SRFlow](https://arxiv.org/abs/2006.14200) is a normalizing-flow based SR technique that eschews GANs entirely in favor
of hooking a SR network to an invertible flow network with the objective of reducing the details of a high-resolution
image into noise indistinguishable from the Gaussian distribution. In the process of doing so, the SRFlow network 
actually trains the underlying SR network to a fairly amazing degree. The end product is a network pair that is adept
at SR, restoration, and extracting high frequency outliers from HQ images.

As of November 2020, this is a new addition to this codebase. The SRFlow code was ported directly from the 
[author's github](https://github.com/andreas128/SRFlow), and is very rough. I'm currently experimenting with trained
models to determine whether it is worth cleaning up.

# Training SRFlow

SRFlow is trained in 3 steps:

1. Pre-train an SR network on a L1 pixel loss. The current state of SRFlow is highly bound to the RRDB architecture
   but that could be changed if desired easily enough. `train_div2k_rrdb_psnr.yml` provides a sample configuration file.
   Search for `<--` in that file, make the required modifications, and run it through the trainer:
   
   `python train.py -opt train_div2k_rrdb_psnr.yml`
   
   The authors recommended training for 200k iterations. I found RRDB converges far sooner than this and stopped my
   training around 100k iterations.
1. Train the first stage of the SRFlow network, where the RRDB network is frozen and the SRFlow layers are "warmed up".
   `train_div2k_srflow.yml` can be used to do this:
   
   `python train.py -opt train_div2k_srflow.yml`
   
   The authors recommend training in this configuration for half of the entire SRFlow training time. Again, I find this
   unnecessary. I saw that the network converges to a stable gaussian NLL on the validation set after ~20k-40k iterations,
   after which I recommend moving to stage 2.
1. Train the second stage of the SRFlow network, where the RRDB network is unfrozen. Do this by editing `train_div2k_srflow.yml`
   and setting `train_RRDB=true`.
   
   After moving to this phase, you should see the gaussian NLL in the validation metrics start to decrease again. This
   is a really cool phase of training, where the gradient pressure from the NLL loss is actively improving your RRDB SR
   network!

# Using SRFlow

SRFlow networks have several interesting potential uses. I'll go over a few of them. I've written a script that you
might find useful for playing with trained SRFlow networks: `scripts/srflow_latent_space_playground.py`. This script does not
take arguments, you will need to modify the code directly. Just a personal preference for these types of tools.

## Super-resolution

Super resolution is performed by feeding an LR image and a latent into the network. The latent is *supposed* to be from
a gaussian distribution sized relative to the LR image, but this depends on how well the SRFlow network could adapt 
itself to your image distribution. For example, I could not get the 8X SR networks to get anywhere near a gaussian; they
always "stored" much of their structural information inside of the latent.

In practice, you can get pretty damned good SR results from this network by simply feeding in zeros for the latents. This
makes the SRFlow show the "mean HQ" representation it has learned for any given LQ image. It is done by setting the
temperature input to the SRFlow network to 0. Here is an injector definition that does just that:
```
  gen_inj:
    type: generator
    generator: generator
    in: [None, lq, None, 0, True]   # <-- '0' here is the temperature.
    out: [gen]
```

You can also accomplish this in `srflow_latent_space_playground.py` by setting the mode to `temperature`.

## Restoration

This was touched on in the SRFlow paper. The authors recommend computing the latents of a corrupted image, then
performing normalization on it. The logic is that the SRFlow network doesn't "know" how to compute corrupted images, so
the process of normalizing the latents will cause it to output the nearest true HR image to the corrupted input image.

In practice, this works sometimes for me, sometimes not. SRFlow has a knack for producing NaNs in the reverse direction
when it encounters LR images and latent pairs that are too far out of the training distribution. This manifests as
black spots or areas of noise in the image.

In practice, what seems to work better is using the above procedure: feed your corrupted image into the SRFlow  network
with a temperature of 0. This will almost always works and generally produces more pleasing results.

You can tinker with the restoration described in the paper in the `srflow_latent_space_playground.py` script by using
the `restore` mode.

## Style transfer

The SRFlow network splits high frequency information from HQ images by design. This high frequency data is encoded in
the latents. These latents can then be fed back into the network with a different LR image to accomplish a sort of 
style transfer. In the paper, the authors transfer fine facial features and it seems to work well. This was hit or miss
for me, but I admittedly did not try to hard (yet). 

You can tinker with latent transfer in the script by using the `latent_transfer` mode. Note that this only does whole-
image latent transfer.

# Notes on validation

My validation runs are my own design. The work by feeding a set of HQ images from your target distribution through the
SRFlow network to produce latents. These latents are then compared to a gaussian distribution and the validation score
is the per-pixel distance from that distribution. I do not compute the log of the loss since this hides fine improvements
at the log levels that this network operates in.