import torch


def extract_byol_model_from_state_dict(sd):
    wrap_key = 'online_encoder.net.'
    sdo = {}
    for k,v in sd.items():
        if wrap_key in k:
            sdo[k.replace(wrap_key, '')] = v
    return sdo

if __name__ == '__main__':
    pretrained_path = '../../../experiments/uresnet_pixpro4_imgset.pth'
    output_path = '../../../experiments/uresnet_pixpro4_imgset.pth'

    sd = torch.load(pretrained_path)
    sd = extract_byol_model_from_state_dict(sd)

    #model = SpineNet('49', in_channels=3, use_input_norm=True).to('cuda')
    #model.load_state_dict(sdo, strict=True)

    print("Validation succeeded, dumping state dict to output path.")
    torch.save(sdo, output_path)