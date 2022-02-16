import torch
import torch.nn as nn
from trainer.networks import register_model
from utils.util import opt_get


def encoder_for_type(type, master_dim, enc_kwargs):
    from x_clip.x_clip import VisionTransformer, TextTransformer
    if type == 'image':
        # xclip_kwargs: image_size, patch_size, channels, depth, heads
        return VisionTransformer(dim=master_dim, **enc_kwargs)
    elif type == 'tokens':
        # xclip_kwargs: num_tokens, max_seq_len, depth, heads
        return TextTransformer(dim=master_dim, **enc_kwargs)
    raise NotImplementedError()


class XClipWrapper(nn.Module):
    def __init__(self,
                 master_dim=512,
                 enc1_type='vision',
                 enc1_kwargs={},
                 enc2_type='text',
                 enc2_kwargs={},
                 mask_seq1_percentage=0,
                 mask_seq2_percentage=0,
                 **xclip_kwargs):
        super().__init__()
        self.mask_seq1_percentage = mask_seq1_percentage
        self.mask_seq2_percentage = mask_seq2_percentage
        enc1 = encoder_for_type(enc1_type, master_dim, enc1_kwargs)
        enc2 = encoder_for_type(enc2_type, master_dim, enc2_kwargs)
        xclip_kwargs['dim_text'] = master_dim
        xclip_kwargs['dim_image'] = master_dim
        xclip_kwargs['dim_latent'] = master_dim
        xclip_kwargs['text_encoder'] = enc1  # The first argument of forward
        xclip_kwargs['image_encoder'] = enc2
        # xclip_kwargs:
        #  use_all_token_embeds
        #  downsample_image_embeds
        #  decoupled_contrastive_learning
        #  extra_latent_projection
        #  use_mlm
        from x_clip import CLIP
        self.clip = CLIP(**xclip_kwargs)

    def forward(self, seq1, seq2, return_loss=False):
        seq1_mask = torch.rand_like(seq1.float()) > self.mask_seq1_percentage
        # TODO: add support for seq2 mask..
        #seq2_mask = torch.rand_like(seq2.float()) > self.mask_seq2_percentage
        return self.clip(seq1, seq2, seq1_mask, return_loss=return_loss)


@register_model
def register_clip(opt_net, opt):
    return XClipWrapper(**opt_get(opt_net, ['kwargs'], {}))

if __name__ == '__main__':
    model = XClipWrapper(enc1_type='tokens', enc2_type='tokens',
                         enc1_kwargs={'num_tokens': 256, 'max_seq_len': 200, 'depth': 8, 'heads': 8},
                         enc2_kwargs={'num_tokens': 8192, 'max_seq_len': 250, 'depth': 8, 'heads': 8})
    loss = model(torch.randint(0,256, (2,200)),  torch.randint(0,8192, (2,250)), True)
    print(loss)