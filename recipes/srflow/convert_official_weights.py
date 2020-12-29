import torch

# Quick script that can be used to convert from pretrained SRFlow weights to the variants used in this repo. The only
# differences between the two is the variable naming conventions used by the RRDBNet. (FWIW this repo is using the
# more up-to-date names that conform to Python standards).

official_weight_file = 'SRFlow_CelebA_8X.pth'
output = 'CelebA_converted.pth'

sd = torch.load(official_weight_file)
sdp = {}
for k,v in sd.items():
    k = k.replace('RRDB.RRDB_trunk', 'RRDB.body')
    k = k.replace('.RDB', '.rdb')
    k = k.replace('trunk_conv.', 'conv_body.')
    k = k.replace('.upconv', '.conv_up')
    k = k.replace('.HRconv', '.conv_hr')
sdp[k] = v
torch.save(sdp, output)
