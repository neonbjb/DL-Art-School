from torch import nn

from trainer.losses import ConfigurableLoss


class FastSpeech2Loss(ConfigurableLoss):
    def __init__(self, opt_loss, env):
        super().__init__(opt_loss, env)
        self.log_d_predicted = opt_loss['log_d_predicted']
        self.log_d_target = opt_loss['log_d_target']
        self.p_predicted = opt_loss['p_predicted']
        self.p_target = opt_loss['p_target']
        self.e_predicted = opt_loss['e_predicted']
        self.e_target = opt_loss['e_target']
        self.mel = opt_loss['mel']
        self.mel_postnet = opt_loss['mel_postnet']
        self.mel_target = opt_loss['mel_target']
        self.src_mask = opt_loss['src_mask']
        self.mel_mask = opt_loss['mel_mask']
        self.last_mel_loss = 0
        self.last_postnet_loss = 0
        self.last_d_loss = 0
        self.last_p_loss = 0
        self.last_e_loss = 0

    def forward(self, _, state):
        log_d_predicted, log_d_target = state[self.log_d_predicted], state[self.log_d_target]
        p_predicted, p_target = state[self.p_predicted], state[self.p_target]
        e_predicted, e_target = state[self.e_predicted], state[self.e_target]
        mel, mel_postnet, mel_target = state[self.mel], state[self.mel_postnet], state[self.mel_target]
        src_mask, mel_mask = ~state[self.src_mask], ~state[self.mel_mask]

        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        p_predicted = p_predicted.masked_select(mel_mask)
        p_target = p_target.masked_select(mel_mask)
        e_predicted = e_predicted.masked_select(mel_mask)
        e_target = e_target.masked_select(mel_mask)

        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)

        self.last_mel_loss = mel_loss
        self.last_postnet_loss = mel_postnet_loss
        self.last_d_loss = d_loss
        self.last_p_loss = p_loss
        self.last_e_loss = e_loss

        return mel_loss + mel_postnet_loss + d_loss + d_loss + .01*p_loss + .01*e_loss

    def extra_metrics(self):
        return {
            'mel_loss': self.last_mel_loss,
            'mel_postnet_loss':  self.last_postnet_loss,
            'd_loss': self.last_d_loss,
            'p_loss': self.last_d_loss,
            'e_loss': self.last_d_loss,
        }
