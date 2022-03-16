import os
import random

import torch
import torchvision
from torch.cuda.amp import autocast

from data.images.multiscale_dataset import build_multiscale_patch_index_map
from trainer.inject import Injector
from trainer.losses import extract_params_from_state
import os.path as osp


def create_progressive_zoom_injector(opt, env):
    type = opt['type']
    if type == 'progressive_zoom_generator':
        return ProgressiveGeneratorInjector(opt, env)
    return None


class ProgressiveGeneratorInjector(Injector):
    def __init__(self, opt, env):
        super(ProgressiveGeneratorInjector, self).__init__(opt, env)
        self.gen_key = opt['generator']
        self.hq_key = opt['hq']  # The key where HQ images are stored.
        self.hq_output_key = opt['hq_output']  # The key where HQ images corresponding with generated images are stored.
        self.input_lq_index = opt['input_lq_index'] if 'input_lq_index' in opt.keys() else 0
        self.output_hq_index = opt['output_hq_index']
        if 'recurrent_output_index' in opt.keys():
            self.recurrent_output_index = opt['recurrent_output_index']
            self.recurrent_index = opt['recurrent_index']
            self.recurrence = True
        else:
            self.recurrence = False
        self.depth = opt['depth']
        self.number_branches = opt['num_branches']  # Number of input branches to randomly choose for generation. This defines the output shape.
        self.multiscale_leaves = build_multiscale_patch_index_map(self.depth)
        self.feed_gen_output_into_input = opt['feed_gen_output_into_input']

    # Given a set of multiscale inputs, selects self.num_branches leaves and produces that many chains of inputs,
    # excluding the base input for efficiency reasons.
    # Output is a list of chains. Each chain is itself a list of links. Each link is MultiscaleTreeNode
    def get_input_chains(self):
        leaves = random.sample(self.multiscale_leaves, self.number_branches)
        chains = []
        for leaf in leaves:
            chain = [leaf]
            node = leaf.parent
            while node.parent is not None:
                chain.insert(0, node)
                node = node.parent
            chains.append(chain)
        return chains

    def feed_forward(self, gen, inputs, results, lq_input, recurrent_input):
        ff_input = inputs.copy()
        ff_input[self.input_lq_index] = lq_input
        if self.recurrence:
            ff_input[self.recurrent_index] = recurrent_input

        with autocast(enabled=self.env['opt']['fp16']):
            gen_out = gen(*ff_input)

        if isinstance(gen_out, torch.Tensor):
            gen_out = [gen_out]
        for i, out_key in enumerate(self.output):
            results[out_key].append(gen_out[i])
        recurrent = None
        if self.recurrence:
            recurrent = gen_out[self.recurrent_output_index]
        return gen_out[self.output_hq_index], recurrent

    def forward(self, state):
        gen = self.env['generators'][self.gen_key]
        inputs = extract_params_from_state(self.input, state)
        lq_inputs = inputs[self.input_lq_index]
        hq_inputs = state[self.hq_key]
        output = self.output
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(self.output, list):
            output = [self.output]
            self.output = output
        results = {}   # A list of outputs produced by feeding each progressive lq input into the generator.
        results_hq = []
        for out_key in output:
            results[out_key] = []

        b, f, h, w = lq_inputs[:, 0].shape
        base_hq_out, base_recurrent = self.feed_forward(gen, inputs, results, lq_inputs[:, 0], None)
        results_hq.append(hq_inputs[:, 0])
        input_chains = self.get_input_chains()
        debug_index = 0
        for chain in input_chains:
            chain_input = [lq_inputs[:, 0]]
            chain_output = [base_hq_out]
            recurrent_hq = base_hq_out
            recurrent = base_recurrent
            for link in chain:  # Remember, `link` is a MultiscaleTreeNode.
                top = int(link.top * h)
                left = int(link.left * w)
                if recurrent is not None:
                    recurrent = torch.nn.functional.interpolate(recurrent[:, :, top:top+h//2, left:left+w//2], scale_factor=2, mode="nearest")
                if self.feed_gen_output_into_input:
                    top *= 2
                    left *= 2
                    lq_input = recurrent_hq[:, :, top:top+h, left:left+w]
                else:
                    lq_input = lq_inputs[:, link.index]
                chain_input.append(lq_input)
                recurrent_hq, recurrent = self.feed_forward(gen, inputs, results, lq_input, recurrent)
                chain_output.append(recurrent_hq)
                results_hq.append(hq_inputs[:, link.index])

            if self.env['step'] % 50 == 0:
                self.produce_progressive_visual_debugs(chain_input, chain_output, debug_index)
                debug_index += 1
        results[self.hq_output_key] = results_hq

        # Results are concatenated into the batch dimension, to allow normal losses to be used against the output.
        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)
        return results


    def produce_progressive_visual_debugs(self, chain_inputs, chain_outputs, it):
        if self.env['rank'] > 0:
            return
        if self.feed_gen_output_into_input:
            lbl = 'generator_recurrent'
        else:
            lbl = 'generator_regular'
        base_path = osp.join(self.env['base_path'], "../../models", "visual_dbg", lbl, str(self.env['step']))
        os.makedirs(base_path, exist_ok=True)
        ind = 1
        for i, o in zip(chain_inputs, chain_outputs):
            torchvision.utils.save_image(i.float(), osp.join(base_path, "%s_%i_input.png" % (it, ind)))
            torchvision.utils.save_image(o.float(), osp.join(base_path, "%s_%i_output.png" % (it, ind)))
            ind += 1
