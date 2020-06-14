import argparse
import options.options as option
from models.networks import define_G
import torch
import torchvision
import torch.nn.functional as F

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../options/use_video_upsample.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    netG = define_G(opt)
    dummyInput = torch.rand(1,3,8,8)

    mode = 'trace'
    if mode == 'torchscript':
        print("Tracing generator network..")
        traced_netG = torch.jit.trace(netG, dummyInput)
        traced_netG.save('../results/ts_generator.zip')

        print(traced_netG.code)
        for i, module in enumerate(traced_netG.RRDB_trunk.modules()):
            print(i, str(module))
    elif mode == 'onnx':
        print("Performing onnx trace")
        input_names = ["lr_input"]
        output_names = ["hr_image"]
        dynamic_axes = {'lr_input': {0: 'batch', 1: 'filters', 2: 'h', 3: 'w'}, 'hr_image': {0: 'batch', 1: 'filters', 2: 'h', 3: 'w'}}

        torch.onnx.export(netG, dummyInput, "../results/gen.onnx", verbose=True, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)
    elif mode == 'trace':
        out = netG.forward(dummyInput)[0]
        print(out.shape)
        # Build the graph backwards.
        graph = build_graph(out, 'output')

def get_unique_id_for_fn(fn):
    return (str(fn).split(" object at ")[1])[:-1]

class GraphNode:
    def __init__(self, fn):
        self.name = (str(fn).split(" object at ")[0])[1:]
        self.fn = fn
        self.children = {}
        self.parents = {}

    def add_parent(self, parent):
        self.parents[get_unique_id_for_fn(parent)] = parent

    def add_child(self, child):
        self.children[get_unique_id_for_fn(child)] = child

class TorchGraph:
    def __init__(self):
        self.tensor_map = {}

    def get_node_for_tensor(self, t):
        return self.tensor_map[get_unique_id_for_fn(t)]

    def init(self, output_tensor):
        self.build_graph_backwards(output_tensor.grad_fn, None)
        # Find inputs
        self.inputs = []
        for v in self.tensor_map.values():
            # Is an input if the parents dict is empty.
            if bool(v.parents):
                self.inputs.append(v)

    def build_graph_backwards(self, fn, previous_fn):
        id = get_unique_id_for_fn(fn)
        if id in self.tensor_map:
            node = self.tensor_map[id]
            node.add_child(previous_fn)
        else:
            node = GraphNode(fn)
            self.tensor_map[id] = node
            # Propagate to children
            for child_fn in fn.next_functions:
                node.add_parent(self.build_graph_backwards(child_fn, fn))
        return node