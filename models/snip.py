"""
I refer some code from here: https://github.com/mi-lad/snip/blob/master/snip.py
"""

import math
import copy, types
import torch
import torch.nn as nn
import torch.nn.functional as F

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias, 
                self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

class SNIP(object):
    def __init__(self, net, device, kappa = 0.9):
        """
            kappa is a sparsity level
            paper use kappa as normal integer, but in here kappa expressed as a percentage
            so, sparsity level = 90 will be kappe = 0.9 in here.
        """
        self.net = net
        self.small_net = copy.deepcopy(net)
        self.device = device
        self.kappa = kappa
        assert kappa <= 1., "kappa value should be range in [0, 1]"

        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight)).to(self.device)
                layer.weight.requires_grad = False
    
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

    def construct_small_network(self, input_x, input_y, kappa = None):
        """
            This function will returns smaller network than you set,
            which is more narrow in linear layerand less depth in conv2d layer

            last layer's output shape should be kept as same
        """
        if isinstance(kappa, float):
            self.kappa = kappa
            assert kappa <= 1., "kappa value should be range in [0, 1]"
        # keep in minds that if channel/vector is removed, upper layer should shrink following to lower layer
        self.net.zero_grad()
        outputs = self.net.forward(input_x)
        loss = F.nll_loss(outputs, input_y)
        loss.backward()

        grads_abs = []
        for layer in self.net.modules():
            # to check sensitivity filter wise, grouping target channels
            if isinstance(layer, nn.Conv2d):
                grads_abs.append(torch.abs(layer.weight_mask.grad).sum(2).sum(2).sum(1))
            elif isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad).sum(1))
            else:
                continue #pass
        
        all_scores = torch.cat(grads_abs) # all flattened
        norm_factor = torch.sum(all_scores)
        # each filter have different number of parameters. Is it fair?
        all_scores.div_(norm_factor)

        num_filter_to_keep = int(len(all_scores) * (1.-self.kappa))
        threshold, _ = torch.topk(all_scores, num_filter_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_filters = []
        for g in grads_abs:
            # for indexing, bool type only supporting after 1.2 version
            keep_filters.append(((g / norm_factor) >= acceptable_score).bool())

        # TODO: contruct new network with removed filters
        prunable_layers = []
        batchnorm_layers = {}
        prev_layer = None
        for i, layer in enumerate(self.small_net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                prunable_layers.append(layer)
                prev_layer = layer
            elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                batchnorm_layers[prev_layer] = layer

        prev_keep_filter = None # input channels
        for i, (layer, keep_filter) in enumerate(zip(prunable_layers, keep_filters)):
            if i == 0:
                # keep all input channels in first layer
                # maybe it will not be zero
                # TODO: need to check when kappa increase
                prev_keep_filter = torch.ones(layer.weight.shape[1]).bool()
            assert (layer.weight.shape[0] == keep_filter.shape[0]), "Channel size is not aligned!"
            # Checked: Kappa increase, some layer totally be zeros. Need to prevent!

            if layer in batchnorm_layers:
                batchnorm_layers[layer].weight = nn.Parameter(batchnorm_layers[layer].weight[keep_filter])
                batchnorm_layers[layer].bias = nn.Parameter(batchnorm_layers[layer].bias[keep_filter])
                batchnorm_layers[layer].running_mean = batchnorm_layers[layer].running_mean[keep_filter]
                batchnorm_layers[layer].running_var = batchnorm_layers[layer].running_var[keep_filter]

            n_params = layer.weight.numel()
            if isinstance(layer, nn.Conv2d):
                layer.weight = nn.Parameter(layer.weight[keep_filter, :, :, :]).to(self.device)
                layer.weight = nn.Parameter(layer.weight[:, prev_keep_filter, :, :]).to(self.device)

            if isinstance(layer, nn.Linear) and isinstance(prunable_layers[i-1], nn.Conv2d):
                out_channel, in_channel = layer.weight.shape
                prev_filter_size = prev_keep_filter.shape[0]
                feature_size = math.sqrt(in_channel / prev_filter_size)
                assert ((feature_size - int(feature_size)) <= 1e-10), "feature size is not aligned!"
                weight = layer.weight.view(out_channel, prev_filter_size, int(feature_size), int(feature_size))
                layer.weight = nn.Parameter(weight[keep_filter, :, :, :])
                layer.weight = nn.Parameter(layer.weight[:, prev_keep_filter, :, :].view(keep_filter.sum(), -1))
            elif isinstance(layer, nn.Linear)and isinstance(prunable_layers[i-1], nn.Linear):
                layer.weight = nn.Parameter(layer.weight[keep_filter, :]).to(self.device)
                layer.weight = nn.Parameter(layer.weight[:, prev_keep_filter]).to(self.device)
            after_n_params = layer.weight.numel()
            print("%dth layer compression rate: %.4f"%(i, (after_n_params/n_params)*100))
            
            layer.bias = nn.Parameter(layer.bias[keep_filter]).to(self.device)
            prev_keep_filter = keep_filter

        n_origin = sum(p.numel() for p in self.net.parameters())
        n_small = sum(p.numel() for p in self.small_net.parameters())

        compressed_rate = (n_small / n_origin) * 100
        print("Compressed rate: %.2f"%compressed_rate)

        return self.small_net, compressed_rate

