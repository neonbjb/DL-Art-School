import torch
import torchvision
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F

def load_img(path):
    im = Image.open(path).convert(mode="RGB")
    return torchvision.transforms.ToTensor()(im)

def save_img(t, path):
    torchvision.utils.save_image(t, path)

img = load_img("pu.jpg")
img = img.unsqueeze(0)

# Reshape image to be multiple of 32
w, h = img.shape[2:]
w = (w // 32) * 32
h = (h // 32) * 32
img = F.interpolate(img, size=(w, h))
print("Input shape:", img.shape)

J_spec = 5

Yl, Yh = DWTForward(J=J_spec, mode='periodization', wave='db3')(img)
print(Yl.shape, [h.shape for h in Yh])

imgLR = F.interpolate(img, scale_factor=.5)
LQYl, LQYh = DWTForward(J=J_spec-1, mode='periodization', wave='db3')(imgLR)
print(LQYl.shape, [h.shape for h in LQYh])

for i in range(J_spec):
    smd = torch.sum(Yh[i], dim=2).cpu()
    save_img(smd, "high_%i.png" % (i,))
save_img(Yl, "lo.png")

'''
Following code reconstructs the image with different high passes cancelled out.
'''
for i in range(J_spec):
    corrupted_im = [y for y in Yh]
    corrupted_im[i] = torch.zeros_like(corrupted_im[i])
    im = DWTInverse(mode='periodization', wave='db3')((Yl, corrupted_im))
    save_img(im, "corrupt_%i.png" % (i,))
im = DWTInverse(mode='periodization', wave='db3')((torch.full_like(Yl, fill_value=torch.mean(Yl)), Yh))
save_img(im, "corrupt_im.png")


'''
Following code reconstructs a hybrid image with the first high pass from the HR and the rest of the data from the LR.
highpass = [Yh[0]] + LQYh
im = DWTInverse(mode='periodization', wave='db3')((LQYl, highpass))
save_img(im, "hybrid_lrhr.png")
save_img(F.interpolate(imgLR, scale_factor=2), "upscaled.png")
'''