# GLEAN

DLAS contains an attempt at implementing [GLEAN](https://ckkelvinchan.github.io/papers/glean.pdf), which performs image
super-resolution guided by pretrained StyleGAN networks. Since this paper is currently closed-source, it was 
implemented entirely on what information I could glean from the paper.

## Training

GLEAN requires a pre-trained StyleGAN network to operate. DLAS currently only has support for StyleGAN2 models, so
you will need to use one of those. The pre-eminent StyleGAN 2 model is the one trained on FFHQ faces, so I will use
that in this training example.

1. Download the ffhq model from [nVidias Drive](https://drive.google.com/drive/folders/1yanUI9m4b4PWzR0eurKNq6JR1Bbfbh6L).
   This repo currently only supports the "-f.pkl" files without further modifications, so choose one of those.
1. Download and extract the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset).
1. Convert the TF model to a Pytorch one supported by DLAS:

   `python scripts/stylegan2/convert_weights_rosinality.py stylegan2-ffhq-config-f.pkl`
   
1. The above conversion script outputs a *.pth file as well as JPG preview of model outputs. Check the JPG to ensure
   the StyleGAN is performing as expected. If so, copy the *.pth file to your experiments/ directory within DLAS.
1. Edit the provided trainer configuration. Find comments starting with '<--' and make changes as indicated.
1. Train the model:

   `python train.py -opt train_ffhq_glean.yml`