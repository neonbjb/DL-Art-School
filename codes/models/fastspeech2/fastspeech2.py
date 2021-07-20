import torch
import torch.nn as nn

from trainer.networks import register_model
from models.fastspeech2.transformer import Encoder, Decoder
from models.fastspeech2.transformer import PostNet

from models.fastspeech2.modules import VarianceAdaptor, SpeakerIntegrator, Embedding
from models.fastspeech2.utils import get_mask_from_lengths
from models.fastspeech2.hparams import defaults as hp


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True, n_spkers=1):
        super(FastSpeech2, self).__init__()

        ### Speaker Embedding Table ###
        self.use_spk_embed = hp.use_spk_embed
        if self.use_spk_embed:
            self.n_spkers = n_spkers
            self.spk_embed_dim = hp.spk_embed_dim
            self.spk_embed_weight_std = hp.spk_embed_weight_std
            self.embed_speakers = Embedding(n_spkers, self.spk_embed_dim, padding_idx=None,
                                            std=self.spk_embed_weight_std)

        ### Encoder, Speaker Integrator, Variance Adaptor, Deocder, Postnet ###
        self.encoder = Encoder()
        if self.use_spk_embed:
            self.speaker_integrator = SpeakerIntegrator()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None,
                max_mel_len=None, speaker_ids=None):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None

        encoder_output = self.encoder(src_seq, src_mask)

        if self.use_spk_embed and speaker_ids is not None:
            spk_embed = self.embed_speakers(speaker_ids)
            encoder_output = self.speaker_integrator(encoder_output, spk_embed)

        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)

        if self.use_spk_embed and speaker_ids is not None:
            variance_adaptor_output = self.speaker_integrator(variance_adaptor_output, spk_embed)

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


@register_model
def register_fastspeech2(opt_net, opt):
    return FastSpeech2()


if __name__ == "__main__":
    # Test

    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))