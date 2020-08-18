# Defines the expected API for a step
class SrGanGeneratorStep:

    def __init__(self, opt_step, opt, netsG, netsD):
        self.step_opt = opt_step
        self.opt = opt
        self.gen = netsG['base']
        self.disc = netsD['base']
        for loss in self.step_opt['losses']:

        # G pixel loss
        if train_opt['pixel_weight'] > 0:
            l_pix_type = train_opt['pixel_criterion']
            if l_pix_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif l_pix_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            self.l_pix_w = train_opt['pixel_weight']
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None


    # Returns all optimizers used in this step.
    def get_optimizers(self):
        pass

    # Returns optimizers which are opting in for default LR scheduling.
    def get_optimizers_with_default_scheduler(self):
        pass

    # Returns the names of the networks this step will train. Other networks will be frozen.
    def get_networks_trained(self):
        pass

    # Performs all forward and backward passes for this step given an input state. All input states are lists or
    # chunked tensors. Use grad_accum_step to derefernce these steps. Return the state with any variables the step
    # exports (which may be used by subsequent steps)
    def do_forward_backward(self, state, grad_accum_step):
        return state

    # Performs the optimizer step after all gradient accumulation is completed.
    def do_step(self):
        pass