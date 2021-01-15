import math
from collections import Counter
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_scheduler_for_name(name, optimizers, scheduler_opt):
    schedulers = []
    for o in optimizers:
        # Hack to support LARC, which wraps an underlying optimizer.
        if hasattr(o, 'optim'):
            o = o.optim

        if name == 'MultiStepLR':
            sched = MultiStepLR_Restart(o, scheduler_opt['gen_lr_steps'],
                                             restarts=scheduler_opt['restarts'],
                                             weights=scheduler_opt['restart_weights'],
                                             gamma=scheduler_opt['lr_gamma'],
                                             clear_state=scheduler_opt['clear_state'],
                                             force_lr=scheduler_opt['force_lr'])
        elif name == 'ProgressiveMultiStepLR':
            sched = ProgressiveMultiStepLR(o, scheduler_opt['gen_lr_steps'],
                                             scheduler_opt['progressive_starts'],
                                             scheduler_opt['lr_gamma'])
        elif name == 'CosineAnnealingLR_Restart':
            sched = CosineAnnealingLR_Restart(
                        o, scheduler_opt['T_period'], scheduler_opt['warmup'], eta_min=scheduler_opt['eta_min'],
                        restarts=scheduler_opt['restarts'], weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError('Scheduler not available')
        schedulers.append(sched)
    return schedulers


# This scheduler is specifically designed to modulate the learning rate of several different param groups configured
# by a generator or discriminator that slowly adds new stages one at a time, e.g. like progressive growing of GANs.
class ProgressiveMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, group_starts, gamma=0.1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.group_starts = group_starts
        super(ProgressiveMultiStepLR, self).__init__(optimizer)

    def get_lr(self):
        group_lrs = []
        assert len(self.optimizer.param_groups) == len(self.group_starts)
        for group, group_start in zip(self.optimizer.param_groups, self.group_starts):
            if self.last_epoch - group_start not in self.milestones:
                group_lrs.append(group['lr'])
            else:
                group_lrs.append(group['lr'] * self.gamma)
        return group_lrs


class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, force_lr=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.force_lr = force_lr
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.force_lr:
            return [group['initial_lr'] for group in self.optimizer.param_groups]
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    # Allow this scheduler to use newly appointed milestones partially through a training run..
    def load_state_dict(self, s):
        milestones_cache = self.milestones
        super(MultiStepLR_Restart, self).load_state_dict(s)
        self.milestones = milestones_cache


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, warmup=0, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.warmup = warmup
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch - self.warmup
        if step <= 0:
            return self.base_lrs
        elif step in self.restarts:
            self.last_restart = step
            self.T_max = self.T_period[self.restarts.index(step) + 1]
            weight = self.restart_weights[self.restarts.index(step)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (step - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (step - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((step - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


if __name__ == "__main__":
    optimizer = torch.optim.Adam([torch.zeros(3, 64, 3, 3)], lr=1e-4, weight_decay=0,
                                 betas=(0.9, 0.99))
    ##############################
    # MultiStepLR_Restart
    ##############################
    ## Original
    lr_steps = [200000, 400000, 600000, 800000]
    restarts = None
    restart_weights = None

    ## two
    lr_steps = [100000, 200000, 300000, 400000, 490000, 600000, 700000, 800000, 900000, 990000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    lr_steps = [
        50000, 100000, 150000, 200000, 240000, 300000, 350000, 400000, 450000, 490000, 550000,
        600000, 650000, 700000, 740000, 800000, 850000, 900000, 950000, 990000
    ]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(optimizer, lr_steps, restarts, restart_weights, gamma=0.5,
                                    clear_state=False)

    ##############################
    # Cosine Annealing Restart
    ##############################
    ## two
    T_period = [500000, 500000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    T_period = [200000, 100000, 200000]
    restarts = [200000, 300000]
    restart_weights = [.5, .25]

    scheduler = CosineAnnealingLR_Restart(optimizer, T_period, warmup=10000, eta_min=1e-8, restarts=restarts,
                                          weights=restart_weights)

    ##############################
    # Draw figure
    ##############################
    N_iter = 500000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_l[i] = current_lr

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Title', fontsize=16, color='k')
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.show()
