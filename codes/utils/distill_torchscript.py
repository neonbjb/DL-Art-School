import argparse
import functools
import torch
import options.options as option
from models.networks import define_G


class TracedModule:
    def __init__(self, idname):
        self.idname = idname
        self.traced_outputs = []
        self.traced_inputs = []


class TorchCustomTrace:
    def __init__(self):
        self.module_name_counter = {}
        self.modules = {}
        self.graph = {}
        self.module_map_by_inputs = {}
        self.module_map_by_outputs = {}
        self.inputs_to_func_output_tuple = {}

    def add_tracked_module(self, mod: torch.nn.Module):
        modname = type(mod).__name__
        if modname not in self.module_name_counter.keys():
            self.module_name_counter[modname] = 0
        self.module_name_counter[modname] += 1
        idname = "%s(%03d)" % (modname, self.module_name_counter[modname])
        self.modules[idname] = TracedModule(idname)
        return idname

    # Only called for nn.Modules since those are the only things we can access. Filling in the gaps will be done in
    # the backwards pass.
    def mem_forward_hook(self, module: torch.nn.Module, inputs, outputs, trace: str, mod_id: str):
        mod = self.modules[mod_id]
        '''
        for li in inputs:
            if type(li) == torch.Tensor:
                li = [li]
            if type(li) == list:
                for i in li:
                    if i.data_ptr() in self.module_map_by_inputs.keys():
                        self.module_map_by_inputs[i.data_ptr()].append(mod)
                    else:
                        self.module_map_by_inputs[i.data_ptr()] = [mod]
        for o in outputs:
            if o.data_ptr() in self.module_map_by_inputs.keys():
                self.module_map_by_inputs[o.data_ptr()].append(mod)
            else:
                self.module_map_by_inputs[o.data_ptr()] = [mod]
        '''
        print(trace)

    def mem_backward_hook(self, inputs, outputs, op):
        if len(inputs) == 0:
            print("No inputs.. %s" % (op,))
        outs = [o.data_ptr() for o in outputs]
        tup = (outs, op)
        #print(tup)
        for li in inputs:
            if type(li) == torch.Tensor:
                li = [li]
            if type(li) == list:
                for i in li:
                    if i.data_ptr() in self.module_map_by_inputs.keys():
                        print("%i: [%s] {%s}" % (i.data_ptr(), op, [n.idname for n in self.module_map_by_inputs[i.data_ptr()]]))
                    if i.data_ptr() in self.inputs_to_func_output_tuple.keys():
                        self.inputs_to_func_output_tuple[i.data_ptr()].append(tup)
                    else:
                        self.inputs_to_func_output_tuple[i.data_ptr()] = [tup]

    def install_hooks(self, mod: torch.nn.Module, trace=""):
        mod_id = self.add_tracked_module(mod)
        my_trace = trace + "->" + mod_id
        # If this module has parameters, it also has a state worth tracking.
        #if next(mod.parameters(recurse=False), None) is not None:
        mod.register_forward_hook(functools.partial(self.mem_forward_hook, trace=my_trace, mod_id=mod_id))

        for m in mod.children():
            self.install_hooks(m, my_trace)

    def install_backward_hooks(self, grad_fn):
        # AccumulateGrad simply pushes a gradient into the specified variable, and isn't useful for the purposes of
        # tracing the graph.
        if grad_fn is None or "AccumulateGrad" in str(grad_fn):
            return
        grad_fn.register_hook(functools.partial(self.mem_backward_hook, op=str(grad_fn)))
        for g, _ in grad_fn.next_functions:
            self.install_backward_hooks(g)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to options YAML file.', default='../../options/train_div2k_pixgan_srg2.yml')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    netG = define_G(opt)
    dummyInput = torch.rand(1,3,32,32)

    mode = 'onnx'
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
                          output_names=output_names, dynamic_axes=dynamic_axes, opset_version=12)
    elif mode == 'memtrace':
        criterion = torch.nn.MSELoss()
        tracer = TorchCustomTrace()
        tracer.install_hooks(netG)
        out, = netG(dummyInput)
        tracer.install_backward_hooks(out.grad_fn)
        target = torch.zeros_like(out)
        loss = criterion(out, target)
        loss.backward()
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