from torch import nn

from trainer.losses import ConfigurableLoss


class Tacotron2Loss(ConfigurableLoss):
    def __init__(self, opt_loss, env):
        super().__init__(opt_loss, env)
        self.mel_target_key = opt_loss['mel_target_key']
        self.mel_output_key = opt_loss['mel_output_key']
        self.mel_output_postnet_key = opt_loss['mel_output_postnet_key']
        self.gate_target_key = opt_loss['gate_target_key']
        self.gate_output_key = opt_loss['gate_output_key']
        self.last_mel_loss = 0
        self.last_gate_loss = 0

    def forward(self, _, state):
        mel_target, gate_target = state[self.mel_target_key], state[self.gate_target_key]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out = state[self.mel_output_key], state[self.mel_output_postnet_key], state[self.gate_output_key]
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        self.last_mel_loss = mel_loss.detach().clone().mean().item()
        self.last_gate_loss = gate_loss.detach().clone().mean().item()
        return mel_loss + gate_loss

    def extra_metrics(self):
        return {
            'mel_loss': self.last_mel_loss,
            'gate_loss':  self.last_gate_loss
        }


class Tacotron2LossRaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_mel_loss = 0
        self.last_gate_loss = 0

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        self.last_mel_loss = mel_loss.detach().clone().mean().item()
        self.last_gate_loss = gate_loss.detach().clone().mean().item()
        return mel_loss + gate_loss

    def extra_metrics(self):
        return {
            'mel_loss': self.last_mel_loss,
            'gate_loss':  self.last_gate_loss
        }