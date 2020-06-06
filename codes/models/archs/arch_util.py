import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from math import sqrt

def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdims=True) + epsilon)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers, return_layers=False):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if return_layers:
        return nn.Sequential(*layers), layers
    else:
        return nn.Sequential(*layers)

class DynamicConv2d(nn.Module):
    def __init__(self, nf_in_per_conv, nf_out_per_conv, kernel_size, stride=1, pads=0, has_bias=True, num_convs=8,
                 att_kernel_size=5, att_pads=2, initial_temperature=1):
        super(DynamicConv2d, self).__init__()

        # Requirements: input filter count is even, and there are more filters than there are sequences to attend to.
        assert nf_in_per_conv % 2 == 0
        assert nf_in_per_conv / 2 > num_convs

        self.nf = nf_out_per_conv
        self.num_convs = num_convs
        self.conv_list = nn.ModuleList([nn.Conv2d(nf_in_per_conv, nf_out_per_conv, kernel_size, stride, pads, bias=has_bias) for _ in range(num_convs)])
        self.attention_conv1 = nn.Conv2d(nf_in_per_conv, int(nf_in_per_conv/2), att_kernel_size, stride, att_pads, bias=True)
        self.att_bn1 = nn.BatchNorm2d(int(nf_in_per_conv/2))
        self.attention_conv2 = nn.Conv2d(int(nf_in_per_conv/2), num_convs, att_kernel_size, 1, att_pads, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.temperature = initial_temperature

    def set_attention_temperature(self, temp):
        self.temperature = temp

    def forward(self, x, output_attention_weights=False):
        # Build up the individual conv components first.
        conv_outputs = []
        for conv in self.conv_list:
            conv_outputs.append(conv.forward(x))
        conv_outputs = torch.stack(conv_outputs, dim=0).permute(1, 3, 4, 2, 0)

        # Now calculate the attention across those convs.
        conv_attention = self.lrelu(self.att_bn1(self.attention_conv1(x)))
        conv_attention = self.attention_conv2(conv_attention).permute(0, 2, 3, 1)
        conv_attention = self.softmax(conv_attention / self.temperature)

        # conv_outputs shape:   (batch, width, height, filters, sequences)
        # conv_attention shape: (batch, width, height, sequences)
        # We want to format them so that we can matmul them together to produce:
        # desired shape:        (batch, width, height, filters)
        attention_result = torch.einsum("...ij,...j->...i", [conv_outputs, conv_attention])

        # Remember to shift the filters back into the expected slot.
        if output_attention_weights:
            return attention_result.permute(0, 3, 1, 2), conv_attention
        else:
            return attention_result.permute(0, 3, 1, 2)

def compute_attention_specificity(att_weights, topk=3):
    att = att_weights.detach()
    vals, indices = torch.topk(att, topk, dim=-1)
    avg = torch.sum(vals, dim=-1)
    avg = avg.flatten().mean()
    return avg.item(), indices.flatten().detach()

class DynamicConvTestModule(nn.Module):
    def __init__(self):
        super(DynamicConvTestModule, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=True)
        self.conv1 = DynamicConv2d(16, 32, 3, stride=2, pads=1, num_convs=4, initial_temperature=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = DynamicConv2d(32, 64, 3, stride=2, pads=1, att_kernel_size=3, att_pads=1, num_convs=8, initial_temperature=10)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = DynamicConv2d(64, 128, 3, stride=2, pads=1, att_kernel_size=3, att_pads=1, num_convs=16, initial_temperature=10)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(128 * 4 * 4, 256)
        self.dense2 = nn.Linear(256, 100)
        self.softmax = nn.Softmax(-1)

    def set_temp(self, temp):
        self.conv1.set_attention_temperature(temp)
        self.conv2.set_attention_temperature(temp)
        self.conv3.set_attention_temperature(temp)

    def forward(self, x):
        x = self.init_conv(x)
        x, att1 = self.conv1(x, output_attention_weights=True)
        x = self.relu(self.bn1(x))
        x, att2 = self.conv2(x, output_attention_weights=True)
        x = self.relu(self.bn2(x))
        x, att3 = self.conv3(x, output_attention_weights=True)
        x = self.relu(self.bn3(x))
        atts = [att1, att2, att3]
        usage_hists = []
        mean = 0
        for a in atts:
            m, u = compute_attention_specificity(a)
            mean += m
            usage_hists.append(u)
        mean /= 3

        x = x.flatten(1)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        # Compute metrics across attention weights.

        return self.softmax(x), mean, usage_hists


class StandardConvTestModule(nn.Module):
    def __init__(self):
        super(StandardConvTestModule, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(16, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(256 * 4 * 4, 256)
        self.dense2 = nn.Linear(256, 100)
        self.softmax = nn.Softmax(-1)

    def set_temp(self, temp):
        pass

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.relu(self.bn3(x))

        x = x.flatten(1)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)

        return self.softmax(x), 0, []

import torch.optim as optim
from torchvision import datasets, models, transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter

def test_dynamic_conv():
    writer = SummaryWriter()
    dataset = datasets.ImageFolder("E:\\data\\cifar-100-python\\images\\train", transforms.Compose([
        transforms.Resize(32, 32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    batch_size = 256
    temperature = 30
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    device = torch.device("cuda:0")
    net = StandardConvTestModule()
    net = net.to(device)
    net.set_temp(temperature)
    initialize_weights(net)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Load state, where necessary.
    '''
    netstate, optimstate = torch.load("test_net.pth")
    net.load_state_dict(netstate)
    optimizer.load_state_dict(optimstate)
    '''

    criterion = nn.CrossEntropyLoss()
    step = 0
    running_corrects = 0
    running_att_mean = 0
    running_att_hist = None
    for e in range(300):
        tq = tqdm.tqdm(loader)
        for batch, labels in tq:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, att_mean, att_usage_hist = net.forward(batch)
            running_att_mean += att_mean
            if running_att_hist is None:
                running_att_hist = att_usage_hist
            else:
                for i in range(len(att_usage_hist)):
                    running_att_hist[i] = torch.cat([running_att_hist[i], att_usage_hist[i]]).flatten()
            loss = criterion(logits, labels)
            loss.backward()

            '''
            if step % 50 == 0:
                c1_grad_avg = sum([m.weight.grad.abs().mean().item() for m in net.conv1.conv_list._modules.values()]) / len(net.conv1.conv_list._modules)
                c1a_grad_avg = (net.conv1.attention_conv1.weight.grad.abs().mean() + net.conv1.attention_conv2.weight.grad.abs().mean()) / 2
                c2_grad_avg = sum([m.weight.grad.abs().mean().item() for m in net.conv2.conv_list._modules.values()]) / len(net.conv2.conv_list._modules)
                c2a_grad_avg = (net.conv2.attention_conv1.weight.grad.abs().mean() + net.conv2.attention_conv2.weight.grad.abs().mean()) / 2
                c3_grad_avg = sum([m.weight.grad.abs().mean().item() for m in net.conv3.conv_list._modules.values()]) / len(net.conv3.conv_list._modules)
                c3a_grad_avg = (net.conv3.attention_conv1.weight.grad.abs().mean() + net.conv3.attention_conv2.weight.grad.abs().mean()) / 2
                writer.add_scalar("c1_grad_avg", c1_grad_avg, global_step=step)
                writer.add_scalar("c2_grad_avg", c2_grad_avg, global_step=step)
                writer.add_scalar("c3_grad_avg", c3_grad_avg, global_step=step)
                writer.add_scalar("c1a_grad_avg", c1a_grad_avg, global_step=step)
                writer.add_scalar("c2a_grad_avg", c2a_grad_avg, global_step=step)
                writer.add_scalar("c3a_grad_avg", c3a_grad_avg, global_step=step)
                '''

            optimizer.step()
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels.data)
            if step % 50 == 0:
                print("Step: %i, Loss: %f, acc: %f, att_mean: %f" % (step, loss.item(), running_corrects / (50.0 * batch_size),
                                                                                   running_att_mean / 50.0))
                writer.add_scalar("Loss", loss.item(), global_step=step)
                writer.add_scalar("Accuracy", running_corrects / (50.0 * batch_size), global_step=step)
                writer.add_scalar("Att Mean", running_att_mean / 50, global_step=step)
                for i in range(len(running_att_hist)):
                    writer.add_histogram("Att Hist %i" % (i,), running_att_hist[i], global_step=step)
                writer.flush()
                running_corrects = 0
                running_att_mean = 0
                running_att_hist = None
            if step % 1000 == 0:
                temperature = max(temperature-1, 1)
                net.set_temp(temperature)
                print("Temperature drop. Now: %i" % (temperature,))
            step += 1
        torch.save((net.state_dict(), optimizer.state_dict()), "test_net_standard.pth")

if __name__ == '__main__':
    test_dynamic_conv()


class ResidualBlock(nn.Module):
    '''Residual block with BN
    ---Conv-BN-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.BN2 = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.BN1(self.conv1(x)))
        out = self.BN2(self.conv2(out))
        return identity + out

class ResidualBlockSpectralNorm(nn.Module):
    '''Residual block with Spectral Normalization.
    ---SpecConv-ReLU-SpecConv-+-
     |________________|
    '''

    def __init__(self, nf, total_residual_blocks):
        super(ResidualBlockSpectralNorm, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = SpectralNorm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.conv2 = SpectralNorm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        initialize_weights([self.conv1, self.conv2], 1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
