import torch
from torch import nn
import models.image_generation.discriminator_vgg_arch as disc
import functools

blacklisted_modules = [nn.Conv2d, nn.ReLU, nn.LeakyReLU, nn.BatchNorm2d, nn.Softmax]
def install_forward_trace_hooks(module, id="base"):
    if type(module) in blacklisted_modules:
        return
    module.register_forward_hook(functools.partial(inject_input_shapes, mod_id=id))
    for name, m in module.named_children():
        cid = "%s:%s" % (id, name)
        install_forward_trace_hooks(m, cid)

def inject_input_shapes(module: nn.Module, inputs, outputs, mod_id: str):
    if len(inputs) == 1 and isinstance(inputs[0], torch.Tensor):
        # Only single tensor inputs currently supported. TODO: fix.
        module._input_shape = inputs[0].shape

def extract_input_shapes(module, id="base"):
    shapes = {}
    if hasattr(module, "_input_shape"):
        shapes[id] = module._input_shape
    for n, m in module.named_children():
        cid = "%s:%s" % (id, n)
        shapes.update(extract_input_shapes(m, cid))
    return shapes

def test_stability(mod_fn, dummy_inputs, device='cuda'):
    base_module = mod_fn().to(device)
    dummy_inputs = dummy_inputs.to(device)
    install_forward_trace_hooks(base_module)
    base_module(dummy_inputs)
    input_shapes = extract_input_shapes(base_module)

    means = {}
    stds = {}
    for i in range(20):
        mod = mod_fn().to(device)
        t_means, t_stds = test_stability_per_module(mod, input_shapes, device)
        for k in t_means.keys():
            if k not in means.keys():
                means[k] = []
                stds[k] = []
            means[k].extend(t_means[k])
            stds[k].extend(t_stds[k])

    for k in means.keys():
        print("%s - mean: %f std: %f" % (k, torch.mean(torch.stack(means[k])),
                                         torch.mean(torch.stack(stds[k]))))

def test_stability_per_module(mod: nn.Module, input_shapes: dict, device='cuda', id="base"):
    means = {}
    stds = {}
    if id in input_shapes.keys():
        format = input_shapes[id]
        mean, std = test_numeric_stability(mod, format, 1, device)
        means[id] = mean
        stds[id] = std
    for name, child in mod.named_children():
        cid = "%s:%s" % (id, name)
        m, s = test_stability_per_module(child, input_shapes, device=device, id=cid)
        means.update(m)
        stds.update(s)
    return means, stds

def test_numeric_stability(mod: nn.Module, format, iterations=50, device='cuda'):
    x = torch.randn(format).to(device)
    means = []
    stds = []
    with torch.no_grad():
        for i in range(iterations):
            x = mod(x)[0]
            measure = x
            means.append(torch.mean(measure).detach())
            stds.append(torch.std(measure).detach())
    return torch.stack(means), torch.stack(stds)


if __name__ == "__main__":
    '''
    test_stability(functools.partial(nsg.NestedSwitchedGenerator,
                                     switch_filters=64,
                                     switch_reductions=[3,3,3,3,3],
                                     switch_processing_layers=[1,1,1,1,1],
                                     trans_counts=[3,3,3,3,3],
                                     trans_kernel_sizes=[3,3,3,3,3],
                                     trans_layers=[3,3,3,3,3],
                                     transformation_filters=64,
                                     initial_temp=10),
                   torch.randn(1, 3, 64, 64),
                   device='cuda')
    '''
    '''
    test_stability(functools.partial(srg.DualOutputSRG,
                                     switch_depth=4,
                                     switch_filters=64,
                                     switch_reductions=4,
                                     switch_processing_layers=2,
                                     trans_counts=8,
                                     trans_kernel_sizes=3,
                                     trans_layers=4,
                                     transformation_filters=64,
                                     upsample_factor=4),
                   torch.randn(1, 3, 32, 32),
                   device='cpu')
    '''
    '''
    test_stability(functools.partial(srg1.ConfigurableSwitchedResidualGenerator,
                                     switch_filters=[32,32,32,32],
                                     switch_growths=[16,16,16,16],
                                     switch_reductions=[4,3,2,1],
                                     switch_processing_layers=[3,3,4,5],
                                     trans_counts=[16,16,16,16,16],
                                     trans_kernel_sizes=[3,3,3,3,3],
                                     trans_layers=[3,3,3,3,3],
                                     trans_filters_mid=[24,24,24,24,24],
                                     initial_temp=10),
                   torch.randn(1, 3, 64, 64),
                   device='cuda')
                   '''
    '''
    test_stability(functools.partial(srg.ConfigurableSwitchedResidualGenerator3,
                                     64, 16),
                   torch.randn(1, 3, 64, 64),
                   device='cuda')
    '''
    test_stability(functools.partial(disc.Discriminator_UNet_FeaOut, 3, 64),
                   torch.randn(1,3,128,128),
                   device='cpu')