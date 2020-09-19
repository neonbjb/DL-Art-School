import torch

def get_experiment_for_name(name):
    return Experiment()


# Experiments are ways to add hooks into the ExtensibleTrainer training process with the intent of reporting the
# inner workings of the process in a custom manner that is unsuitable for addition elsewhere.
class Experiment:
    def before_step(self, opt, step_name, env, nets_to_train, pre_state):
        pass

    def before_optimize(self, state):
        pass

    def after_optimize(self, state):
        pass

    def get_log_data(self):
        pass


class ModelParameterDepthTrackerMetrics(Experiment):
    # Subclasses should implement these two methods:
    def get_network_and_step_names(self):
        # Subclasses should return the network being debugged and the step name it is trained in. return: (net, stepname)
        pass
    def get_layers_to_debug(self, env, net, state):
        # Subclasses should populate self.layers with a list of per-layer nn.Modules here.
        pass


    def before_step(self, opt, step_name, env, nets_to_train, pre_state):
        self.net, step = self.get_network_and_step_names()
        self.activate = self.net in nets_to_train and step == step_name and self.step_num % opt['logger']['print_freq'] == 0
        if self.activate:
            layers = self.get_layers_to_debug(env, env['networks'][self.net], pre_state)
            self.params = []
            for l in layers:
                lparams = []
                for k, v in env['networks'][self.net].named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        lparams.append(v)
                self.params.append(lparams)

    def before_optimize(self, state):
        self.cached_params = []
        for l in self.params:
            lparams = []
            for p in l:
                lparams.append(p.value().cpu())
            self.cached_params.append(lparams)

    def after_optimize(self, state):
        # Compute the abs mean difference across the params.
        self.layer_means = []
        for l, lc in zip(self.params, self.cached_params):
            sum = torch.tensor(0)
            for p, pc in zip(l, lc):
                sum += torch.abs(pc - p.value().cpu())
            self.layer_means.append(sum / len(l))

    def get_log_data(self):
        return {'%s_layer_update_means_histogram' % (self.net,): self.layer_means}


class DiscriminatorParameterTracker(ModelParameterDepthTrackerMetrics):
    def get_network_and_step_names(self):
        return "feature_discriminator", "feature_discriminator"

    def get_layers_to_debug(self, env, net, state):
        return [net.ref_head.conv0_0,
                net.ref_head.conv0_1,
                net.ref_head.conv1_0,
                net.ref_head.conv1_1,
                net.ref_head.conv2_0,
                net.ref_head.conv2_1,
                net.ref_head.conv3_0,
                net.ref_head.conv3_1,
                net.ref_head.conv4_0,
                net.ref_head.conv4_1,
                net.linear1,
                net.output_linears]