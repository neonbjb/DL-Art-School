import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SequenceDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(SequenceDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

# Input into this block is of shape (sequence, filters, width, height)
# Output is (attention_hidden_size, width, height)
class ConvAttentionBlock(nn.Module):

    def __init__(self, planes, attention_hidden_size=8, query_conv=conv1x1, key_conv=conv1x1, value_conv=conv1x1):
        super(ConvAttentionBlock, self).__init__()
        self.query_conv_dist = SequenceDistributed(query_conv(planes, attention_hidden_size))
        self.key_conv_dist = SequenceDistributed(key_conv(planes, attention_hidden_size))
        self.value_conv_dist = value_conv(planes, attention_hidden_size)
        self.hidden_size = attention_hidden_size

    def forward(self, x):
        # All values come out of this with the shape (batch, sequence, hidden, width, height)
        query = self.query_conv_dist(x)
        key = self.key_conv_dist(x)
        value = self.value_conv_dist(x)

        # Permute to (batch, width, height, sequence, hidden)
        query = query.permute(0, 3, 4, 1, 2)
        key = key.permute(0, 3, 4, 1, 2)
        value = value.permute(0, 3, 4, 1, 2)

        # Perform attention operation.
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(self.hidden_size)
        scores = torch.softmax(scores, dim=-1)
        result = torch.matmul(scores, value)

        # Collapse out the sequence dim.
        result = torch.sum(result, dim=-2)

        # Permute back to (batch, hidden, width, height)
        result = result.permute(0, 3, 1, 2)
        return result
