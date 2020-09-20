import models.steps.injectors as injectors


# Uses a generator to synthesize a sequence of images from [in] and injects the results into a list [out]
# All results are checkpointed for memory savings. Recurrent inputs are also detached before being fed back into
# the generator.
class RecurrentImageGeneratorSequenceInjector(injectors.Injector):
    def __init__(self, opt, env):
        super(RecurrentImageGeneratorSequenceInjector, self).__init__(opt, env)

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        new_state = {}
        results = []
        recurrent_input = torch.zeros_like(state[self.input][0])
        for input in state[self.input]:
            result = checkpoint(gen, input, recurrent_input)
            results.append(result)
            recurrent_input = result.detach()

        new_state = {self.output: results}
        return new_state


class ImageFlowInjector(injectors.Injector):
    def __init__(self, opt, env):
        # Requires building this custom cuda kernel. Only require it if explicitly needed.
        from models.networks.layers.resample2d_package.resample2d import Resample2d
        super(ImageFlowInjector, self).__init__(opt, env)
        self.resample = Resample2d()

    def forward(self, state):
        return self.resample(state[self.opt['in']], state[self.opt['flow']])
