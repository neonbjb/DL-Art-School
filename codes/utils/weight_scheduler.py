import math
from matplotlib import pyplot as plt

# Base class for weight schedulers. Holds weight at a fixed initial value.
class WeightScheduler:
    def __init__(self, initial_weight):
        self.initial_weight = initial_weight

    def get_weight_for_step(self, step):
        return self.initial_weight


class LinearDecayWeightScheduler(WeightScheduler):
    def __init__(self, initial_weight, steps_to_decay, lower_bound, initial_step=0):
        super(LinearDecayWeightScheduler, self).__init__(initial_weight)
        self.steps_to_decay = steps_to_decay
        self.lower_bound = lower_bound
        self.initial_step = initial_step
        self.decrease_per_step = (initial_weight - lower_bound) / self.steps_to_decay

    def get_weight_for_step(self, step):
        step = step - self.initial_step
        if step < 0:
            return self.initial_weight
        return max(self.lower_bound, self.initial_weight - step * self.decrease_per_step)


class SinusoidalWeightScheduler(WeightScheduler):
    def __init__(self, upper_weight, lower_weight, period_steps, initial_step=0):
        super(SinusoidalWeightScheduler, self).__init__(upper_weight)
        self.center = (upper_weight + lower_weight) / 2
        self.amplitude = (upper_weight - lower_weight) / 2
        self.period = period_steps
        self.initial_step = initial_step

    def get_weight_for_step(self, step):
        step = step - self.initial_step
        if step < 0:
            return self.initial_weight
        # Use cosine because it starts at y=1 for x=0.
        return math.cos(step * math.pi * 2 / self.period) * self.amplitude + self.center


def get_scheduler_for_opt(opt):
    if opt['type'] == 'fixed':
        return WeightScheduler(opt['weight'])
    elif opt['type'] == 'linear_decay':
        return LinearDecayWeightScheduler(opt['initial_weight'], opt['steps'], opt['lower_bound'], opt['start_step'])
    elif opt['type'] == 'sinusoidal':
        return SinusoidalWeightScheduler(opt['upper_weight'], opt['lower_weight'], opt['period'], opt['start_step'])
    else:
        raise NotImplementedError


# Do some testing.
if __name__ == "__main__":
    #sched = SinusoidalWeightScheduler(1, .1, 50, 10)
    sched = LinearDecayWeightScheduler(10, 5000, .9, 2000)

    x = []
    y = []
    for s in range(8000):
        x.append(s)
        y.append(sched.get_weight_for_step(s))
    plt.plot(x, y)
    plt.show()