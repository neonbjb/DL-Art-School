import models.steps.injectors as injectors
import torch


# Uses a generator to synthesize a sequence of images from [in] and injects the results into a list [out]
# Images are fed in sequentially forward and back, resulting in len([out])=2*len([in])-1 (last element is not repeated).
# All computation is done with torch.no_grad().
class RecurrentImageGeneratorSequenceInjector(injectors.Injector):
    def __init__(self, opt, env):
        super(RecurrentImageGeneratorSequenceInjector, self).__init__(opt, env)

    def forward(self, state):
        gen = self.env['generators'][self.opt['generator']]
        results = []
        with torch.no_grad():
            recurrent_input = torch.zeros_like(state[self.input][0])
            # Go forward in the sequence first.
            for input in state[self.input]:
                recurrent_input = gen(input, recurrent_input)
                results.append(recurrent_input)

            # Now go backwards, skipping the last element (it's already stored in recurrent_input)
            it = reversed(range(len(results) - 1))
            for i in it:
                recurrent_input = gen(results[i], recurrent_input)
                results.append(recurrent_input)

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
