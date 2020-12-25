import munch
import torch

from trainer.networks import register_model


@register_model
def register_flownet2(opt_net):
    from models.flownet2.models import FlowNet2
    ld = 'load_path' in opt_net.keys()
    args = munch.Munch({'fp16': False, 'rgb_max': 1.0, 'checkpoint': not ld})
    netG = FlowNet2(args)
    if ld:
        sd = torch.load(opt_net['load_path'])
        netG.load_state_dict(sd['state_dict'])