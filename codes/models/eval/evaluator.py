# Base class for an evaluator, which is responsible for feeding test data through a model and evaluating the response.
class Evaluator:
    def __init__(self, model, opt_eval, env):
        self.model = model.module if hasattr(model, 'module') else model
        self.opt = opt_eval
        self.env = env

    def perform_eval(self):
        return {}
