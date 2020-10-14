import torch


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = True
        with torch.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        print("Backpropping")
        input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        return (None, None) + input_grads

from models.archs.arch_util import ConvGnSilu
import torch.nn as nn
if __name__ == "__main__":
    model = nn.Sequential(ConvGnSilu(3, 64, 3, norm=False),
                          ConvGnSilu(64, 3, 3, norm=False)
                          )
    model.train()
    seed = torch.randn(1,3,32,32)
    recurrent = seed
    outs = []
    for i in range(10):
        args = (recurrent, ) + tuple(model.parameters())
        recurrent = CheckpointFunction.apply(model, 1, *args)
        outs.append(recurrent)

    l = nn.L1Loss()(recurrent, torch.randn(1,3,32,32))
    l.backward()