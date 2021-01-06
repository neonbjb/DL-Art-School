import torch

from models.spinenet_arch import SpineNet

if __name__ == '__main__':
    pretrained_path = '../../experiments/resnet_byol_diffframe_115k.pth'
    output_path = '../../experiments/resnet_byol_diffframe_115k_.pth'

    wrap_key = 'online_encoder.net.'
    sd = torch.load(pretrained_path)
    sdo = {}
    for k,v in sd.items():
        if wrap_key in k:
            sdo[k.replace(wrap_key, '')] = v

    #model = SpineNet('49', in_channels=3, use_input_norm=True).to('cuda')
    #model.load_state_dict(sdo, strict=True)

    print("Validation succeeded, dumping state dict to output path.")
    torch.save(sdo, output_path)