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

all_layers = []
def _remove_sequential(network):
    for layer in network.children():
        if list(layer.children()) != []: # if sequential layer, apply recursively to layers in sequential layer
            _remove_sequential(layer)
        elif not isinstance(layer, nn.Sequential) and (isinstance(layer, nn.Conv2d) or
                isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm2d) or 
                isinstance(layer, nn.BatchNorm1d)): # if leaf node and have weight, add it to list
            all_layers.append(layer)

def is_same_tensor(tensor1, tensor2):
    if tensor1.data.ne(tensor2.data).sum() > 0:
        return False
    return True

class SNIP(object):
    def __init__(self, net, device, kappa = 0.9):
        """
            kappa is a sparsity level
            paper use kappa as normal integer, but in here kappa expressed as a percentage
            so, sparsity level = 90 will be kappe = 0.9 in here.
        """
        self.net = net
        self.small_net = copy.deepcopy(net)
        #self.small_net = copy.copy(net)
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

    def construct_small_network(self, input_x, input_y, kappa = None, loss = None, image_size = None):
        """
            This function will returns smaller network than you set,
            which is more narrow in linear layerand less depth in conv2d layer

            last layer's output shape should be kept as same
        """
        if isinstance(kappa, float):
            self.kappa = kappa
            assert kappa <= 1., "kappa value should be range in [0, 1]"
        # keep in minds that if channel/vector is removed, upper layer should shrink following to lower layer
        if loss is None:
            self.net.zero_grad()
            outputs = self.net.forward(input_x)
            loss = F.nll_loss(outputs, input_y)
            loss.backward()
        else:
            loss.backward()

        global all_layers
        all_layers = []
        _remove_sequential(self.net) # fill_all_layers
        
        grads_abs = []
        for layer in all_layers:
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

        # TODO: filter align
        if input_x != None:
            parsed_graph, params_dict = self._graph_parsing(self.small_net, input_x)
        else:

            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            elif not (isinstance(image_size, tuple) or isinstance(image_size, list)):
                raise ValueError("Error in image_size:", image_size)

            input_x = torch.ones((1, 3, image_size[0], image_size[1])).cuda()
            parsed_graph, params_dict = self._graph_parsing(self.small_net, input_x)

        all_layers = []
        _remove_sequential(self.small_net) # fill_all_layers

        # prunable_layers are in params_dict as a value
        # need to check change of parameter in dict.value affects to small_net
        n_origin = sum(p.numel() for p in self.small_net.parameters())

        for i, layer in enumerate(all_layers):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                for key, value in params_dict.items(): # ordered
                    if isinstance(value, nn.Conv2d) or isinstance(value, nn.Linear) or isinstance(value, nn.BatchNorm2d) :
                        continue
                    if value.shape == layer.weight.shape and is_same_tensor(layer.weight, value):
                        params_dict[key] = layer
                        break
                # I am not sure that this is safe way to align
                if i != len(all_layers)-1:
                    if isinstance(all_layers[i+1], nn.BatchNorm2d) or isinstance(all_layers[i+1], nn.BatchNorm2d):
                        if isinstance(all_layers[i].bias, torch.Tensor):
                            params_dict[str(int(key)+2)] = all_layers[i+1]
                        else:
                            params_dict[str(int(key)+1)] = all_layers[i+1]
            else:
                pass

        # TODO: filter align
        layer_filter = self._filter_align(parsed_graph, params_dict, all_layers, keep_filters)
        self._network_reconstruction(layer_filter, all_layers)

        n_small = sum(p.numel() for p in self.small_net.parameters())

        comp_rate = (1. - (n_small / n_origin)) * 100
        print("Compression rate: %.2f"%comp_rate)

        return self.small_net, comp_rate

    def _network_reconstruction(self, layer_filter, all_layers):
        for i, layer in enumerate(all_layers):
            keep_filter, prev_keep_filter = layer_filter[layer]
            n_params = layer.weight.numel()
            if isinstance(layer, nn.Linear) and (i-1) >= 0:
                j = 1
                prev_layer = all_layers[i-j]
                while not (isinstance(prev_layer, nn.Conv2d) or isinstance(prev_layer, nn.Linear)):
                    j += 1
                    prev_layer = all_layers[i-j]
                if isinstance(prev_layer, nn.Conv2d):
                    out_channel, in_channel = layer.weight.shape
                    prev_filter_size = prev_keep_filter.shape[0]
                    feature_size = math.sqrt(in_channel / prev_filter_size)
                    assert ((feature_size - int(feature_size)) <= 1e-10), "feature size is not aligned!"
                    weight = layer.weight.view(out_channel, prev_filter_size, int(feature_size), int(feature_size))
                    layer.weight = nn.Parameter(weight[keep_filter, :, :, :]).to(self.device)
                    layer.weight = nn.Parameter(layer.weight[:, prev_keep_filter, :, :].view(keep_filter.sum(), -1)).to(self.device)
                elif isinstance(prev_layer, nn.Linear):
                    layer.weight = nn.Parameter(layer.weight[keep_filter, :]).to(self.device)
                    layer.weight = nn.Parameter(layer.weight[:, prev_keep_filter]).to(self.device)

                if isinstance(layer.bias, torch.Tensor):
                    layer.bias = nn.Parameter(layer.bias[keep_filter]).to(self.device)
            elif isinstance(layer, nn.Conv2d):
                layer.weight = nn.Parameter(layer.weight[keep_filter, :, :, :]).to(self.device)
                if layer.weight.shape[1] == 1:
                    pass
                else:
                    layer.weight = nn.Parameter(layer.weight[:, prev_keep_filter, :, :]).to(self.device)
                
                if layer.groups > 1: # for depthwise conv
                    layer.groups = layer.weight.shape[0]

                if isinstance(layer.bias, torch.Tensor):
                    layer.bias = nn.Parameter(layer.bias[keep_filter]).to(self.device)
            elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.weight = nn.Parameter(layer.weight[keep_filter]).to(self.device)
                layer.bias = nn.Parameter(layer.bias[keep_filter]).to(self.device)
                layer.running_mean = layer.running_mean[keep_filter].to(self.device)
                layer.running_var = layer.running_var[keep_filter].to(self.device)
            else:
                pass
            after_n_params = layer.weight.numel()
            print("%dth layer compression rate: %.4f%%"%(i, (1.-(after_n_params/n_params))*100))

    def _filter_align(self, parsed_graph, params_dict, all_layers, keep_filters):
        # parsed_graph should ordered dict,
        # and should aligned with keep_filter and parsed_graph(conv and linear)
        layer_filter = dict()

        idx = 0
        input_nodes = []
        # check input nodes to prevent deletion of output node
        for key, value in parsed_graph.items():
            input_nodes.extend(value['inputs'])
        input_nodes = set(input_nodes)

        for key, value in parsed_graph.items(): # parsed_graph should ordered dict,
            # key is layer number
            layer_type = value['type']
            if layer_type == "Conv" or layer_type == "Gemm": # Gemm == Linear
                current_layer = params_dict[value['weights'][0]]

                if 'input' in value['inputs'][0]:
                    prev_keep_filter = torch.ones(current_layer.weight.shape[1]).bool()
                else:
                    prev_node = parsed_graph[value['inputs'][0]] # init
                    while prev_node['weights'] == []:
                        # it will stop if loop meet conv/linear/batchnorm layer
                        prev_node = parsed_graph[prev_node['inputs'][0]]
                    
                    prev_layer = params_dict[prev_node['weights'][0]]
                    prev_keep_filter = layer_filter[prev_layer][0]

                #if key not in input_nodes or idx >= 23: # exception
                if key not in input_nodes:
                    # keep all connections for output node
                    keep_filter = torch.ones(current_layer.weight.shape[0]).bool()
                elif current_layer.out_channels != current_layer.weight.shape[1] and current_layer.groups > 1: # for depthwise conv
                    keep_filter = prev_keep_filter
                else:
                    keep_filter = keep_filters[idx]
                layer_filter[current_layer] = (keep_filter, prev_keep_filter)

                idx += 1
            elif layer_type == "BatchNormalization":
                current_layer = params_dict[value['weights'][0]]

                prev_node = parsed_graph[value['inputs'][0]]
                while prev_node['weights'] == []:
                    prev_node = parsed_graph[prev_node['inputs'][0]]

                prev_layer = params_dict[prev_node['weights'][0]]
                keep_filter = layer_filter[prev_layer][0]
                layer_filter[current_layer] = (keep_filter, None)
            elif layer_type == "Add": # maybe shortcut
                branch1, branch2 = value['inputs'][0], value['inputs'][1]
                prev_node1 = parsed_graph[branch1]
                prev_node2 = parsed_graph[branch2]
                all_nodes = []
                all_branches = []
                def _track_all_connections(node):
                    if node['weights'] != []:
                        all_nodes.append(node)
                        if not (node['type'] == 'Conv' or node['type'] == 'Gemm'):
                            _track_all_connections(parsed_graph[node['inputs'][0]])
                    else:
                        if node['inputs'] == []:
                            pass
                        else:
                            prev_node = parsed_graph[node['inputs'][0]]
                            if prev_node['type'] == 'Add':
                                all_branches.append(node['inputs'][0])
                                _track_all_connections(parsed_graph[prev_node['inputs'][0]])
                                _track_all_connections(parsed_graph[prev_node['inputs'][1]])
                            else:
                                _track_all_connections(parsed_graph[prev_node['inputs'][0]])

                _track_all_connections(prev_node1)
                _track_all_connections(prev_node2)

                # need to union all filters into keep_filter
                keep_filter = layer_filter[params_dict[all_nodes[0]['weights'][0]]][0].cpu()
                for node in all_nodes:
                    keep_filter += layer_filter[params_dict[node['weights'][0]]][0].cpu()

                for node in all_nodes:
                    layer = params_dict[node['weights'][0]]
                    layer_filter[layer] = (keep_filter, layer_filter[layer][1])

                # update intermediate layer between shortcut and next shortcut
                for branch in all_branches:
                    node_num = int(branch)
                    while not (parsed_graph[str(node_num)]['type'] == 'Conv' or parsed_graph[str(node_num)] == 'Gemm'):
                        node_num += 1
                        if str(node_num-1) not in parsed_graph[str(node_num)]['inputs']:
                            raise ValueError("Need to fix branch tracing parts")
                    layer = params_dict[parsed_graph[str(node_num)]['weights'][0]]
                    layer_filter[layer] = (layer_filter[layer][0], keep_filter)
                    if parsed_graph[str(node_num+1)]['type'] == "BatchNormalization":
                        batch_layer = params_dict[parsed_graph[str(node_num+1)]['weights'][0]]
                        layer_filter[batch_layer] = (layer_filter[batch_layer][0], keep_filter)
            
                for node in all_nodes:
                    layer = params_dict[node['weights'][0]]

                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        prev_node = parsed_graph[node['inputs'][0]] # init
                        while prev_node['weights'] == []:
                            # it will stop if loop meet conv/linear/batchnorm layer
                            prev_node = parsed_graph[prev_node['inputs'][0]]
                    
                        prev_layer = params_dict[prev_node['weights'][0]]
                        prev_keep_filter = layer_filter[prev_layer][0]
                        layer_filter[layer] = (layer_filter[layer][0], prev_keep_filter)
            elif layer_type == "Concat" and value['weights'] != []: # exception
                if value['weights'][0] not in parsed_graph:
                    weight = params_dict[value['weights'][0]]
                    layer_filter[weight] = (weight.shape[0], None)

            elif value['weights'] != []: # custom layer case...?
                if value['weights'][0] not in parsed_graph:
                    weight = params_dict[value['weights'][0]]

                    prev_node = parsed_graph[str(int(key)-1)]
                    while prev_node['weights'] == []:
                        # it will stop if loop meet conv/linear/batchnorm layer
                        prev_node = parsed_graph[prev_node['inputs'][0]]
                    
                    prev_layer = params_dict[prev_node['weights'][0]]
                    prev_keep_filter = layer_filter[prev_layer][0]
                    keep_filter = prev_keep_filter
                    layer_filter[weight] = (keep_filter, None)
                    weight = weight[keep_filter]
            else:
                pass
        return layer_filter

    def _graph_parsing(self, net, input_x):
        # dummy input is enough
        from torch.onnx import utils
        import re

        graph, params_dict, torch_out = utils._model_to_graph(net, input_x)
        parsed_graph = {}
        # parsed_graph format = { %number: {'type':layer.type, 'input':%number, 'weights': list()}
        type_matcher = re.compile('::(\w+)')
        node_matcher = re.compile('%(\d+|input.\d)')
        # group(0) is all searched regex

        nodes = str(graph).split('\n')
        for node in nodes:
            node = node.strip()
            if '::' in node:
                # parse number, type, input, weights
                type_m = type_matcher.findall(node)
                layer_type = type_m[0]
                node_m = node_matcher.findall(node)
                output_node = node_m[0]

                if layer_type == "Conv" or layer_type == "BatchNormalization" or layer_type == "Gemm":
                    input_nodes = [node_m[1]]
                    # what about multiple input instead weight(in given layer types?)

                    weights = []
                    for i in range(2, len(node_m)):
                        weights.append(node_m[i])
                else:
                    input_nodes = []
                    weights = []
                    for i in range(1, len(node_m)):
                        if node_m[i] in params_dict:
                            weights.append(node_m[i])
                        else:
                            input_nodes.append(node_m[i])
                parsed_graph[output_node] = {'type': layer_type, 'inputs': input_nodes, 'weights': weights}
        return parsed_graph, params_dict
