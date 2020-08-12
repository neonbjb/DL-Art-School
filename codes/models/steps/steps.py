

def create_step(opt_step):
    pass


# Defines the expected API for a step
class base_step:
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