from math import sqrt
import torch
from munch import munchify
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, Flatten

from models.arch_util import ConvGnSilu
from models.diffusion.unet_diffusion import UNetModel, AttentionPool2d
from models.tacotron2.layers import ConvNorm, LinearNorm
from models.tacotron2.hparams import create_hparams
from models.tacotron2.tacotron2 import Prenet, Attention, Encoder
from trainer.networks import register_model
from models.tacotron2.taco_utils import get_mask_from_lengths
from utils.util import opt_get, checkpoint



class WavDecoder(nn.Module):
    def __init__(self, dec_channels, K_ms=40, sample_rate=8000, dropout_probability=.1):
        super().__init__()
        self.dec_channels = dec_channels
        self.K = int(sample_rate * (K_ms/1000))
        self.clarifier = UNetModel(image_size=self.K,
                                   in_channels=1,
                                   model_channels=dec_channels // 4,  # This is a requirement to enable to load the embedding produced by the decoder into the unet model.
                                   out_channels=2,  # 2 channels: eps_pred and variance_pred
                                   num_res_blocks=2,
                                   attention_resolutions=(8,),
                                   dims=1,
                                   dropout=.1,
                                   channel_mult=(1,2,4,8),
                                   use_raw_y_as_embedding=True)
        assert self.K % 64 == 0  # Otherwise the UNetModel breaks.
        self.pre_rnn = nn.Sequential(ConvGnSilu(1,32,kernel_size=5,convnd=nn.Conv1d),
                                     ConvGnSilu(32,64,kernel_size=5,stride=4,convnd=nn.Conv1d),
                                     ConvGnSilu(64,128,kernel_size=5,stride=4,convnd=nn.Conv1d),
                                     ConvGnSilu(128,256,kernel_size=5,stride=4,convnd=nn.Conv1d),
                                     ConvGnSilu(256,dec_channels,kernel_size=1,convnd=nn.Conv1d),
                                     AttentionPool2d(self.K//64,dec_channels,dec_channels//4))
        self.attention_rnn = nn.LSTMCell(dec_channels*2, dec_channels)
        self.attention_layer = Attention(dec_channels, dec_channels, dec_channels)
        self.decoder_rnn = nn.LSTMCell(dec_channels*2, dec_channels, 1)
        self.linear_projection = LinearNorm(dec_channels*2, self.dec_channels)
        self.gate_layer = LinearNorm(self.dec_channels*2, 1, bias=True, w_init_gain='sigmoid')
        self.dropout_probability = dropout_probability

    def chunk_wav(self, wav):
        wavs = list(torch.split(wav, self.K, dim=-1))
        # Pad the last chunk as needed.
        padding_needed = self.K - wavs[-1].shape[-1]
        if padding_needed > 0:
            wavs[-1] = F.pad(wavs[-1], (0,padding_needed))

        wavs = torch.stack(wavs, dim=1)  # wavs.shape = (b,s,K) where s=decoder sequence length
        return wavs, padding_needed
 
    def prepare_decoder_inputs(self, inp):
        # inp.shape = (b,s,K) chunked waveform.
        b,s,K = inp.shape
        first_frame = torch.zeros(b,1,K).to(inp.device)
        x = torch.cat([first_frame, inp[:,:-1]], dim=1)  # It is now aligned for teacher forcing.
        return x

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.dec_channels).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.dec_channels).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.dec_channels).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.dec_channels).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.dec_channels).zero_())

        self.memory = memory
        self.processed_memory = checkpoint(self.attention_layer.memory_layer, memory)
        self.mask = mask

    def teardown_states(self):
        self.attention_hidden = None
        self.attention_cell = None
        self.decoder_hidden = None
        self.decoder_cell = None
        self.attention_weights = None
        self.attention_weights_cum = None
        self.attention_context = None
        self.memory = None
        self.processed_memory = None

    def produce_context(self, decoder_input):
        """ Produces a context and a stop token prediction using the built-in RNN.
        PARAMS
        ------
        decoder_input: prior diffusion step that has been resolved.

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.dropout_probability, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = checkpoint(self.attention_layer, self.attention_hidden, self.memory,
                                                                    self.processed_memory, attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.dropout_probability, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = checkpoint(self.linear_projection, decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def recombine(self, diffusion_eps, gate_outputs, alignments, padding_added):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments, dim=1).repeat(1, self.K, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs, dim=1).repeat(1, self.K)

        b,s,_,K = diffusion_eps.shape
        # (B, S, 2, K) -> (B, 2, S*K)
        diffusion_eps = diffusion_eps.permute(0,2,1,3).reshape(b, 2, s*K)

        return diffusion_eps[:,:,:-padding_added], gate_outputs[:,:-padding_added], alignments[:,:-padding_added]

    def forward(self, wav_noised, wav_real, timesteps, text_enc, memory_lengths):
        '''
        Performs a training forward pass with the given data.
        :param wav_noised: (b,n) diffused waveform tensor on the interval [-1,1]
        :param wav_real: (b,n) actual waveform tensor
        :param text_enc: (b,e) embedding post-encoder with e=self.dec_channels
        '''

        # Start by splitting up the provided waveforms into discrete segments.
        wav_noised, padding_added = self.chunk_wav(wav_noised)
        wav_real, _ = self.chunk_wav(wav_real)
        wav_real = self.prepare_decoder_inputs(wav_real)
        b,s,K = wav_real.shape
        wav_real = checkpoint(self.pre_rnn, wav_real.reshape(b*s,1,K)).reshape(b,s,self.dec_channels)

        self.initialize_decoder_states(text_enc, mask=~get_mask_from_lengths(memory_lengths))
        decoder_contexts, gate_outputs, alignments = [], [], []
        while len(decoder_contexts) < wav_real.size(1):
            decoder_input = wav_real[:, len(decoder_contexts)]
            dec_context, gate_output, attention_weights = self.produce_context(decoder_input)
            decoder_contexts += [dec_context.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
        self.teardown_states()

        # diffusion_inputs and wavs needs to have the sequence and batch dimensions combined, and needs a channel dimension
        diffusion_emb = torch.stack(decoder_contexts, dim=1)
        b,s,c = diffusion_emb.shape
        diffusion_emb = diffusion_emb.reshape(b*s,c)
        wav_noised = wav_noised.reshape(b*s,1,self.K)
        diffusion_eps = self.clarifier(wav_noised, timesteps.repeat(s), diffusion_emb).reshape(b,s,2,self.K)
        # Recombine diffusion outputs across the sequence into a single prediction.
        diffusion_eps, gate_outputs, alignments = self.recombine(diffusion_eps, gate_outputs, alignments, padding_added)
        return diffusion_eps, gate_outputs, alignments


class WaveTacotron2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = WavDecoder(hparams.encoder_embedding_dim)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask_fill = outputs[0].shape[-1]
            mask = ~get_mask_from_lengths(output_lengths, mask_fill)
            mask = mask.unsqueeze(1).repeat(1,2,1)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[0] = outputs[0].unsqueeze(1)  # Re-add channel dimension.
            outputs[1].data.masked_fill_(mask[:,0], 1e3)  # gate energies

        return outputs

    def forward(self, wavs_diffused, wavs_corrected, timesteps, text_inputs, text_lengths, output_lengths):
        # Squeeze the channel dimension out of the input wavs - we only handle single-channel audio here.
        wavs_diffused = wavs_diffused.squeeze(dim=1)
        wavs_corrected = wavs_corrected.squeeze(dim=1)

        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = checkpoint(self.encoder, embedded_inputs, text_lengths)
        eps_pred, gate_outputs, alignments = self.decoder(
            wavs_diffused, wavs_corrected, timesteps, encoder_outputs, memory_lengths=text_lengths)

        return self.parse_output([eps_pred, gate_outputs, alignments], output_lengths)


@register_model
def register_diffusion_wavetron(opt_net, opt):
    hparams = create_hparams()
    hparams.update(opt_net)
    hparams = munchify(hparams)
    return WaveTacotron2(hparams)


if __name__ == '__main__':
    tron = register_diffusion_wavetron({}, {})
    out = tron(wavs_diffused=torch.randn(2, 1, 22000),
               wavs_corrected=torch.randn(2, 1, 22000),
               timesteps=torch.LongTensor([555, 543]),
               text_inputs=torch.randint(high=24, size=(2,12)),
               text_lengths=torch.tensor([12, 12]),
               output_lengths=torch.tensor([21995]))
    print([o.shape for o in out])