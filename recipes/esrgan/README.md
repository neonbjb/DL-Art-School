# Training super-resolution networks with ESRGAN

[SRGAN](https://arxiv.org/abs/1609.04802) is a landmark SR technique. It is quickly approaching "seminal" status because of how many papers
use some or all of the techniques originally introduced in this paper. [ESRGAN](https://arxiv.org/abs/1809.00219) is a followup
paper by the same authors which strictly improves the results of SRGAN.

After considerable trial and error, I recommend an additional set of modifications to ESRGAN to
improve training performance and reduce artifacts:

*  Gradient penalty loss on the discriminator keeps the discriminator gradients to the generator in check.
*  Adding noise of 1/255 to the discriminator prevents it from using the fixed input range of HR images for discrimination. (e.g. - natural HR images can only have values in increments of 1/255, while the generator has continuous outputs. The discriminator can cheat by using this fact.)
*  Adding GroupNorm to the discriminator layers. This further stabilizes gradients without the downsides of BatchNorm.
*  Adding a translational loss to the generator term. This loss works by computing using the generator to compute two HQ images 
   during each training pass from random sub-patches of the original image. A L1 loss is then computed across the shared
   region of the two outputs with a very high gain. I found this to be tremendously helpful in reducing GAN artifacts
   as it forces the generator to be self-consistent.
*  Use a vanilla GAN. The ESRGAN paper promotes the use of RAGAN but I found its effect on result qualit to be minimal 
   with the above modifications. In some cases, it can actually be harmful because it drives strange training
   dynamics on the discriminator. For example, I've observed the output of the discriminator to sometimes
   "explode" when using RAGAN because it does not force a fixed output value. It is also more computationally expensive
   to compute.
   
The examples below have all of these modifications added. I've also provided a reference file that
should be closer to the original ESRGAN implementation, `train_div2k_esrgan_reference.yml`.

## Training ESRGAN

DLAS can train and use ESRGAN models end-to-end. These docs will show you how.

### Dataset Preparation

Start by assembling your dataset. The ESRGAN paper uses the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and 
[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets. These include a small set of high-resolution
images. ESRGAN is trained on small sub-patches of those images. Generate these patches using the instructions found
in 'Generating a chunked dataset' [here](https://github.com/neonbjb/DL-Art-School/blob/gan_lab/codes/data/README.md).

Consider creating a validation set at the same time. These can just be a few medium-resolution, high-quality
images. DLAS will downsample them for you and send them through your network for validation.

### Training the model

Use the train_div2k_esrgan.yml configuration file in this directory as a template to train your
ESRGAN. Search the file for `<--` to find options that will need to be adjusted for your installation.

Train with:
`python train.py -opt train_div2k_esrgan.yml`

Note that this configuration trains an RRDB network with an L1 pixel loss only for the first 100k
steps. I recommend you save the model at step 100k (this is done by default, just copy the file
out of the experiments/train_div2k_esrgan/models directory once it hits step 100k) so that you 
do not need to repeat this training in future experiments.

## Using an ESRGAN model

### Image SR

You can apply a pre-trained ESRGAN model against a set of images using the code in `test.py`.
Documentation for this script is forthcoming but basically you feed it your training configuration
file with the `pretrain_model_generator` option set properly and your folder with test images
pointed to in the datasets section in lieu of the validation set.

### Video SR

I've put together a script that strips a video into its constituent frames, applies an ESRGAN
model to each frame one a time, then recombines the frames back into videos (without sound).
You will need to use ffmpeg to stitch the videos back together and add sound, but this is
trivial.

This script is called `process_video.py` and it takes a special configuration file. A sample
config is provided in `rrdb_process_video.yml` in this directory. Further documentation on this
procedure is forthcoming.

Fun fact: the foundations of DLAS lie in the (now defunct) MMSR github repo, which was
primarily an implementation of ESRGAN.
