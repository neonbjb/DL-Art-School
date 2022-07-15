import itertools
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision

from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from models.diffusion.unet_diffusion import TimestepBlock
from models.lucidrains.x_transformers import Encoder, Attention, RMSScaleShiftNorm, RotaryEmbedding, \
    FeedForward
from trainer.networks import register_model
from utils.util import checkpoint, print_network, load_audio


class TimestepRotaryEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, rotary_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, rotary_emb)
            else:
                x = layer(x, rotary_emb)
        return x


class SubBlock(nn.Module):
    def __init__(self, inp_dim, contraction_dim, heads, dropout, use_conv):
        super().__init__()
        self.attn = Attention(inp_dim, out_dim=contraction_dim, heads=heads, dim_head=contraction_dim//heads, causal=False, dropout=dropout)
        self.attnorm = nn.LayerNorm(contraction_dim)
        self.use_conv = use_conv
        if use_conv:
            self.ff = nn.Conv1d(inp_dim+contraction_dim, contraction_dim, kernel_size=3, padding=1)
        else:
            self.ff = FeedForward(inp_dim+contraction_dim, dim_out=contraction_dim, mult=2, dropout=dropout)
        self.ffnorm = nn.LayerNorm(contraction_dim)

    def forward(self, x, rotary_emb):
        ah, _, _, _ = checkpoint(self.attn, x, None, None, None, None, None, rotary_emb)
        ah = F.gelu(self.attnorm(ah))
        h = torch.cat([ah, x], dim=-1)
        hf = checkpoint(self.ff, h.permute(0,2,1) if self.use_conv else h)
        hf = F.gelu(self.ffnorm(hf.permute(0,2,1) if self.use_conv else hf))
        h = torch.cat([h, hf], dim=-1)
        return h


class ConcatAttentionBlock(TimestepBlock):
    def __init__(self, trunk_dim, contraction_dim, time_embed_dim, cond_dim_in, cond_dim_hidden, heads, dropout, cond_projection=True, use_conv=False):
        super().__init__()
        self.prenorm = RMSScaleShiftNorm(trunk_dim, embed_dim=time_embed_dim, bias=False)
        if cond_projection:
            self.tdim = trunk_dim+cond_dim_hidden
            self.cond_project = nn.Linear(cond_dim_in, cond_dim_hidden)
        else:
            self.tdim = trunk_dim
        self.block1 = SubBlock(self.tdim, contraction_dim, heads, dropout, use_conv)
        self.block2 = SubBlock(self.tdim+contraction_dim*2, contraction_dim, heads, dropout, use_conv)
        self.out = nn.Linear(contraction_dim*4, trunk_dim, bias=False)
        self.out.weight.data.zero_()

    def forward(self, x, cond, timestep_emb, rotary_emb):
        h = self.prenorm(x, norm_scale_shift_inp=timestep_emb)
        if hasattr(self, 'cond_project'):
            cond = self.cond_project(cond)
            h = torch.cat([h, cond], dim=-1)
        h = self.block1(h, rotary_emb)
        h = self.block2(h, rotary_emb)
        h = self.out(h[:,:,self.tdim:])
        return h + x


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 cond_dim,
                 embedding_dim,
                 time_embed_dim,
                 attn_blocks=6,
                 num_attn_heads=8,
                 dropout=.1,
                 do_checkpointing=False,
                 time_proj=True):
        super().__init__()
        self.init = nn.Conv1d(cond_dim, embedding_dim, kernel_size=1)
        self.time_proj = time_proj
        if time_proj:
            self.time_proj = nn.Linear(time_embed_dim, embedding_dim)
        self.attn = Encoder(
                dim=embedding_dim,
                depth=attn_blocks,
                heads=num_attn_heads,
                ff_dropout=dropout,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                zero_init_branch_output=True,
                ff_mult=2,
                do_checkpointing=do_checkpointing
            )
        self.dim = embedding_dim

    def forward(self, x, time_emb):
        h = self.init(x).permute(0,2,1)
        if self.time_proj:
            time_enc = self.time_proj(time_emb)
            h = torch.cat([time_enc.unsqueeze(1), h], dim=1)
        h = self.attn(h).permute(0,2,1)
        return h


class TransformerDiffusionWithPointConditioning(nn.Module):
    def __init__(
            self,
            in_channels=256,
            out_channels=512,  # mean and variance
            model_channels=1024,
            contraction_dim=256,
            time_embed_dim=256,
            num_layers=8,
            rotary_emb_dim=32,
            input_cond_dim=1024,
            num_heads=8,
            dropout=0,
            time_proj=True,
            new_cond=False,
            use_fp16=False,
            checkpoint_conditioning=True,  # This will need to be false for DDP training. :(
            regularization=False,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.regularization = regularization
        self.new_cond = new_cond

        self.inp_block = conv_nd(1, in_channels, model_channels, 3, 1, 1)
        self.conditioning_encoder = ConditioningEncoder(256, model_channels, time_embed_dim, do_checkpointing=checkpoint_conditioning, time_proj=time_proj)

        self.time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.unconditioned_embedding = nn.Parameter(torch.randn(1,1,model_channels))
        self.rotary_embeddings = RotaryEmbedding(rotary_emb_dim)
        self.layers = TimestepRotaryEmbedSequential(*[ConcatAttentionBlock(model_channels,
                                                                           contraction_dim,
                                                                           time_embed_dim,
                                                                           cond_dim_in=input_cond_dim,
                                                                           cond_dim_hidden=input_cond_dim//2,
                                                                           heads=num_heads,
                                                                           dropout=dropout,
                                                                           cond_projection=(k % 3 == 0),
                                                                           use_conv=(k % 3 != 0),
                                                                           ) for k in range(num_layers)])
        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )
        self.debug_codes = {}

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable([lyr.block1.attn.parameters() for lyr in self.layers]))
        attn2 = list(itertools.chain.from_iterable([lyr.block2.attn.parameters() for lyr in self.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff.parameters() for lyr in self.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff.parameters() for lyr in self.layers]))
        blkout_layers = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'rotary_embeddings': list(self.rotary_embeddings.parameters()),
            'out': list(self.out.parameters()),
            'x_proj': list(self.inp_block.parameters()),
            'layers': list(self.layers.parameters()),
            'time_embed': list(self.time_embed.parameters()),
            'conditioning_encoder': list(self.conditioning_encoder.parameters()),
        }
        return groups

    def process_conditioning(self, conditioning_input, time_emb, N, cond_start, cond_left, cond_right):
        if self.training and self.regularization:
            # frequency regularization
            fstart = random.randint(0, conditioning_input.shape[1] - 1)
            fclip = random.randint(1, min(conditioning_input.shape[1]-fstart, 16))
            conditioning_input[:,fstart:fstart+fclip] = 0
            # time regularization
            for k in range(1, random.randint(2, 4)):
                tstart = random.randint(0, conditioning_input.shape[-1] - 1)
                tclip = random.randint(1, min(conditioning_input.shape[-1]-tstart, 10))
                conditioning_input[:,:,tstart:tstart+tclip] = 0

        if cond_left is None and self.new_cond:
            assert cond_start > 20 and (cond_start+N+20 <= conditioning_input.shape[-1])
            cond_left = conditioning_input[:,:,:cond_start]
            left_pt = -1
            cond_right = conditioning_input[:,:,cond_start+N:]
            right_pt = 0
        elif cond_left is None:
            assert conditioning_input.shape[-1] - cond_start - N >= 0, f'Some sort of conditioning misalignment, {conditioning_input.shape[-1], cond_start, N}'
            cond_pre = conditioning_input[:,:,:cond_start]
            cond_aligned = conditioning_input[:,:,cond_start:N+cond_start]
            cond_post = conditioning_input[:,:,N+cond_start:]

            # Break up conditioning input into two random segments aligned with the input.
            MIN_MARGIN = 8
            assert N > (MIN_MARGIN*2+4), f"Input size too small. Was {N} but requires at least {MIN_MARGIN*2+4}"
            break_pt = random.randint(2, N-MIN_MARGIN*2-2) + MIN_MARGIN
            cond_left = cond_aligned[:,:,:break_pt]
            cond_right = cond_aligned[:,:,break_pt:]

            if self.training:
                # Drop out a random amount of the aligned data. The network will need to figure out how to reconstruct this.
                to_remove_left = random.randint(1, cond_left.shape[-1]-MIN_MARGIN)
                cond_left = cond_left[:,:,:-to_remove_left]
                to_remove_right = random.randint(1, cond_right.shape[-1]-MIN_MARGIN)
                cond_right = cond_right[:,:,to_remove_right:]

            # Concatenate the _pre and _post back on.
            left_pt = cond_start
            right_pt = cond_right.shape[-1]
            cond_left = torch.cat([cond_pre, cond_left], dim=-1)
            cond_right = torch.cat([cond_right, cond_post], dim=-1)
        else:
            left_pt = -1
            right_pt = 0

        # Propagate through the encoder.
        cond_left_enc = self.conditioning_encoder(cond_left, time_emb)
        cs = cond_left_enc[:,:,left_pt]
        cond_right_enc = self.conditioning_encoder(cond_right, time_emb)
        ce = cond_right_enc[:,:,right_pt]

        cond_enc = torch.cat([cs.unsqueeze(-1), ce.unsqueeze(-1)], dim=-1)
        cond = F.interpolate(cond_enc, size=(N,), mode='linear', align_corners=True).permute(0,2,1)
        return cond

    def forward(self, x, timesteps, conditioning_input=None, cond_left=None, cond_right=None, conditioning_free=False, cond_start=0):
        unused_params = []

        time_emb = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        if conditioning_free:
            cond = self.unconditioned_embedding
            cond = cond.repeat(1,x.shape[-1],1)
        else:
            cond = self.process_conditioning(conditioning_input, time_emb, x.shape[-1], cond_start, cond_left, cond_right)
            # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
            if self.training and self.unconditioned_percentage > 0:
                unconditioned_batches = torch.rand((cond.shape[0], 1, 1),
                                                   device=cond.device) < self.unconditioned_percentage
                cond = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(cond.shape[0], 1, 1), cond)
            unused_params.append(self.unconditioned_embedding)

        with torch.autocast(x.device.type, enabled=self.enable_fp16):
            x = self.inp_block(x).permute(0,2,1)

            rotary_pos_emb = self.rotary_embeddings(x.shape[1]+1, x.device)
            for layer in self.layers:
                x = checkpoint(layer, x, cond, time_emb, rotary_pos_emb)

        x = x.float().permute(0,2,1)
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out

    def before_step(self, step):
        scaled_grad_parameters = list(itertools.chain.from_iterable([lyr.out.parameters() for lyr in self.layers])) + \
                                 list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.layers]))

        # Scale back the gradients of the blkout and prenorm layers by a constant factor. These get two orders of magnitudes
        # higher gradients. Ideally we would use parameter groups, but ZeroRedundancyOptimizer makes this trickier than
        # directly fiddling with the gradients.
        for p in scaled_grad_parameters:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad *= .2


@register_model
def register_tfdpc5(opt_net, opt):
    return TransformerDiffusionWithPointConditioning(**opt_net['kwargs'])


def test_cheater_model():
    clip = torch.randn(2, 256, 350)
    cl = torch.randn(2, 256, 646)
    ts = torch.LongTensor([600, 600])

    # For music:
    model = TransformerDiffusionWithPointConditioning(in_channels=256, out_channels=512, model_channels=1024,
                                                      contraction_dim=512, num_heads=8, num_layers=32, dropout=0,
                                                      unconditioned_percentage=.4, checkpoint_conditioning=False,
                                                      regularization=True, new_cond=True)
    print_network(model)
    #for cs in range(276,cl.shape[-1]-clip.shape[-1]):
    #    o = model(clip, ts, cl, cond_start=cs)
    pg = model.get_grad_norm_parameter_groups()
    def prmsz(lp):
        sz = 0
        for p in lp:
            q = 1
            for s in p.shape:
                q *= s
            sz += q
        return sz
    for k, v in pg.items():
        print(f'{k}: {prmsz(v)/1000000}')


def test_conditioning_splitting_logic():
    ts = torch.LongTensor([600])
    class fake_conditioner(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, t, _):
            print(t[:,0])
            return t
    model = TransformerDiffusionWithPointConditioning(in_channels=256, out_channels=512, model_channels=1024,
                                                        contraction_dim=512, num_heads=8, num_layers=15, dropout=0,
                                                        unconditioned_percentage=.4)
    model.conditioning_encoder = fake_conditioner()
    BASEDIM=30
    for x in range(BASEDIM+1, BASEDIM+20):
        start = random.randint(0,x-BASEDIM)
        cl = torch.arange(1, x+1, 1).view(1,1,-1).float().repeat(1,256,1)
        print("Effective input: " + str(cl[0, 0, start:BASEDIM+start]))
        res = model.process_conditioning(cl, ts, BASEDIM, start, None)
        print("Result: " + str(res[0,:,0]))
        print()


def inference_tfdpc5_with_cheater():
    with torch.no_grad():
        os.makedirs('results/tfdpc_v3', exist_ok=True)

        # length = 40 * 22050 // 256 // 16
        samples = {'electronica1': load_audio('Y:\\split\\yt-music-eval\\00001.wav', 22050),
                   'electronica2': load_audio('Y:\\split\\yt-music-eval\\00272.wav', 22050),
                   'e_guitar': load_audio('Y:\\split\\yt-music-eval\\00227.wav', 22050),
                   'creep': load_audio('Y:\\separated\\bt-music-3\\[2007] MTV Unplugged (Live) (Japan Edition)\\05 - Creep [Cover On Radiohead]\\00001\\no_vocals.wav', 22050),
                   'rock1': load_audio('Y:\\separated\\bt-music-3\\2016 - Heal My Soul\\01 - Daze Of The Night\\00000\\no_vocals.wav', 22050),
                   'kiss': load_audio('Y:\\separated\\bt-music-3\\KISS (2001) Box Set CD1\\02 Deuce (Demo Version)\\00000\\no_vocals.wav', 22050),
                   'purp': load_audio('Y:\\separated\\bt-music-3\\Shades of Deep Purple\\11 Help (Alternate Take)\\00001\\no_vocals.wav', 22050),
                   'western_stars': load_audio('Y:\\separated\\bt-music-3\\Western Stars\\01 Hitch Hikin\'\\00000\\no_vocals.wav', 22050),
                   'silk': load_audio('Y:\\separated\\silk\\MonstercatSilkShowcase\\890\\00007\\no_vocals.wav', 22050),
                   'long_electronica': load_audio('C:\\Users\\James\\Music\\longer_sample.wav', 22050),}
        for k, sample in samples.items():
            sample = sample.cuda()
            length = sample.shape[0]//256//16

            model = TransformerDiffusionWithPointConditioning(in_channels=256, out_channels=512, model_channels=1024,
                                                              contraction_dim=512, num_heads=8, num_layers=12, dropout=0,
                                                              use_fp16=False, unconditioned_percentage=0).eval().cuda()
            model.load_state_dict(torch.load('x:/dlas/experiments/train_music_cheater_gen_v3/models/59000_generator_ema.pth'))

            from trainer.injectors.audio_injectors import TorchMelSpectrogramInjector
            spec_fn = TorchMelSpectrogramInjector({'n_mel_channels': 256, 'mel_fmax': 11000, 'filter_length': 16000, 'true_normalization': True,
                                                        'normalize': True, 'in': 'in', 'out': 'out'}, {}).cuda()
            ref_mel = spec_fn({'in': sample.unsqueeze(0)})['out']
            from trainer.injectors.audio_injectors import MusicCheaterLatentInjector
            cheater_encoder = MusicCheaterLatentInjector({'in': 'in', 'out': 'out'}, {}).cuda()
            ref_cheater = cheater_encoder({'in': ref_mel})['out']

            from models.diffusion.respace import SpacedDiffusion
            from models.diffusion.respace import space_timesteps
            from models.diffusion.gaussian_diffusion import get_named_beta_schedule
            diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [128]), model_mean_type='epsilon',
                                   model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                                   conditioning_free=True, conditioning_free_k=1)

            # Conventional decoding method:
            gen_cheater = diffuser.ddim_sample_loop(model, (1,256,length), progress=True, model_kwargs={'true_cheater': ref_cheater})

            # Guidance decoding method:
            #mask = torch.ones_like(ref_cheater)
            #mask[:,:,15:-15] = 0
            #gen_cheater = diffuser.p_sample_loop_with_guidance(model, ref_cheater, mask, model_kwargs={'true_cheater': ref_cheater})

            # Just decode the ref.
            #gen_cheater = ref_cheater

            from models.audio.music.transformer_diffusion12 import TransformerDiffusionWithCheaterLatent
            diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [32]), model_mean_type='epsilon',
                                   model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                                   conditioning_free=True, conditioning_free_k=1)
            wrap = TransformerDiffusionWithCheaterLatent(in_channels=256, out_channels=512, model_channels=1024,
                                                         contraction_dim=512, prenet_channels=1024, input_vec_dim=256,
                                                         prenet_layers=6, num_heads=8, num_layers=16, new_code_expansion=True,
                                                         dropout=0, unconditioned_percentage=0).eval().cuda()
            wrap.load_state_dict(torch.load('x:/dlas/experiments/train_music_diffusion_tfd_cheater_from_scratch/models/56500_generator_ema.pth'))
            cheater_to_mel = wrap.diff
            gen_mel = diffuser.ddim_sample_loop(cheater_to_mel, (1,256,gen_cheater.shape[-1]*16), progress=True,
                                             model_kwargs={'codes': gen_cheater.permute(0,2,1)})
            torchvision.utils.save_image((gen_mel + 1)/2, f'results/tfdpc_v3/{k}.png')

            from utils.music_utils import get_mel2wav_v3_model
            m2w = get_mel2wav_v3_model().cuda()
            spectral_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [32]), model_mean_type='epsilon',
                                   model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                                   conditioning_free=True, conditioning_free_k=1)
            from trainer.injectors.audio_injectors import denormalize_mel
            gen_mel_denorm = denormalize_mel(gen_mel)
            output_shape = (1,16,gen_mel_denorm.shape[-1]*256//16)
            gen_wav = spectral_diffuser.ddim_sample_loop(m2w, output_shape, model_kwargs={'codes': gen_mel_denorm})
            from trainer.injectors.audio_injectors import pixel_shuffle_1d
            gen_wav = pixel_shuffle_1d(gen_wav, 16)

            torchaudio.save(f'results/tfdpc_v3/{k}.wav', gen_wav.squeeze(1).cpu(), 22050)
            torchaudio.save(f'results/tfdpc_v3/{k}_ref.wav', sample.unsqueeze(0).cpu(), 22050)

if __name__ == '__main__':
    test_cheater_model()
    #test_conditioning_splitting_logic()
    #inference_tfdpc5_with_cheater()
