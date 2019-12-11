# SNIP-pytorch
Evaluate sensitivity of channel/vector connections to decrease width/depth of network. Which will accelerate inference speed and reduce storage without any other modules or sparse multiplications

This implementation is based on [SNIP paper](https://arxiv.org/abs/1810.02340) and calculate group sensitivity in Linear and Conv2d.

If you have any question or find any issue, please [let me know](../..//issues)!

## Requirements



## Usage

Below are some usage examples, to apply your own model. For more details, please read train.py. 


```python
# This is general case
import torch
Import SNIP

device = torch.device(args.device)
model = build_model(model_name, num_classes, pretrained)
model = model.to(device)
snip = SNIP(model, device, kappa = 0.5)

train_loader = data_loader() # generate input x and y
input_x, input_y = next(iter(train_loader))
input_x, input_y = input_x.to(device), input_y.to(device)

compressed_model, comp_rate = snip.construct_small_network(input_x, input_y)
# Now you can use compressed_model to train
```

## Performances

To check benchmarks for original network performances, you can see from [here, others' implementation](https://github.com/kuangliu/pytorch-cifar)

[ ] Will update performances soon!


## Reference
  - [Another pytorch implementation of SNIP](https://github.com/mi-lad/snip)
   - I refered forward overriding and getting mask parts(snip.py), nice implementation!
  - [Official code, Tensorflow implementation](https://github.com/namhoonlee/snip-public)
  - [SNIP: Single-shot Network Pruning based on Connection Sensitivity (ICLR 2019)](https://arxiv.org/abs/1810.02340)
