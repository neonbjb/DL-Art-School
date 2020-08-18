def create_generator_loss(opt_loss):
    pass


class GeneratorLoss:
    def __init__(self, opt):
        self.opt = opt

    def get_loss(self, var_L, var_H, var_Gen, extras=None):