# Contains implementations from the Mixture of Experts paper and Switch Transformers


# Implements KeepTopK where k=1 from mixture of experts paper.
import torch
import torch.nn as nn

from models.switched_conv.switched_conv_hard_routing import RouteTop1
from trainer.losses import ConfigurableLoss


class KeepTop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        mask = torch.nn.functional.one_hot(input.argmax(dim=1), num_classes=input.shape[1]).permute(0,3,1,2)
        input[mask != 1] = -float('inf')
        ctx.save_for_backward(mask)
        return input

    @staticmethod
    def backward(ctx, grad):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        mask = ctx.saved_tensors
        grad_input = grad.clone()
        grad_input[mask != 1] = 0
        return grad_input

class MixtureOfExperts2dRouter(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.wnoise = nn.Parameter(torch.zeros(1, num_experts, 1, 1))
        self.wg = nn.Parameter(torch.zeros(1, num_experts, 1, 1))

    def forward(self, x):
        wg = x * self.wg
        wnoise = nn.functional.softplus(x * self.wnoise)
        H = wg + torch.randn_like(x) * wnoise

        # Produce the load-balancing loss.
        eye = torch.eye(self.num_experts, device=x.device).view(1, self.num_experts, self.num_experts, 1, 1)
        mask = torch.abs(1 - eye)
        b, c, h, w = H.shape
        ninf = torch.zeros_like(eye)
        ninf[eye == 1] = -float('inf')
        H_masked = H.view(b, c, 1, h,
                          w) * mask + ninf  # ninf is necessary because otherwise torch.max() will not pick up negative numbered maxes.
        max_excluding = torch.max(H_masked, dim=2)[0]

        # load_loss and G are stored as local members to facilitate their use by hard routing regularization losses.
        # this is a risky op - it can easily result in memory leakage. Clients *must* use self.reset() below.
        self.load_loss = torch.erf((wg - max_excluding) / wnoise)
        # self.G = nn.functional.softmax(KeepTop1.apply(H), dim=1)  The paper proposes this equation, but performing a softmax on a Top-1 per the paper results in zero gradients into H, so:
        self.G = RouteTop1.apply(nn.functional.softmax(H, dim=1))  # This variant can route gradients downstream.

        return self.G

    # Retrieve the locally stored loss values and delete them from membership (so as to not waste memory)
    def reset(self):
        G, load = self.G, self.load_loss
        del self.G
        del self.load_loss
        return G, load


# Loss that finds instances of MixtureOfExperts2dRouter in the given network and extracts their custom losses.
class MixtureOfExpertsLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.routers = []  # This is filled in during the first forward() pass and cached from there.
        self.first_forward_encountered = False
        self.load_weight = opt['load_weight']
        self.importance_weight = opt['importance_weight']

    def forward(self, net, state):
        if not self.first_forward_encountered:
            for m in net.modules():
                if isinstance(m, MixtureOfExperts2dRouter):
                    self.routers.append(m)
            self.first_forward_encountered = True

        l_importance = 0
        l_load = 0
        for r in self.routers:
            G, L = r.reset()
            l_importance += G.var().square()
            l_load += L.var().square()
        return l_importance * self.importance_weight + l_load * self.load_weight


class SwitchTransformersLoadBalancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = SwitchNorm(8, accumulator_size=256)

    def forward(self, x):
        self.soft = self.norm(nn.functional.softmax(x, dim=1))
        self.hard = RouteTop1.apply(self.soft)  # This variant can route gradients downstream.
        return self.hard

    def reset(self):
        soft, hard = self.soft, self.hard
        del self.soft, self.hard
        return soft, hard


class SwitchTransformersLoadBalancingLoss(ConfigurableLoss):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        self.routers = []  # This is filled in during the first forward() pass and cached from there.
        self.first_forward_encountered = False

    def forward(self, net, state):
        if not self.first_forward_encountered:
            for m in net.modules():
                if isinstance(m, SwitchTransformersLoadBalancer):
                    self.routers.append(m)
            self.first_forward_encountered = True

        loss = 0
        for r in self.routers:
            soft, hard = r.reset()
            N = hard.shape[1]
            h_mean = hard.mean(dim=[0,2,3])
            s_mean = soft.mean(dim=[0,2,3])
            loss += torch.dot(h_mean, s_mean) * N
        return loss