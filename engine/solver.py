import torch

def build_optimizer(args, model):
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.lr * 2 # BIAS_LR_FACTOR
            weight_decay = 0 # no weight decay for bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    print(" [*] optimizer is SGD")
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    """
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer

def build_lr_scheduler(args, optimizer):
    
    if 'multistep' in args.scheduler_type.lower():
        print(" [*] lr_scheduler is multistepLR")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decaysteps, gamma=args.lr_anneal)
    else:
        raise NotImplementedError(" [!] Not implemented scheduler yet")

    return lr_scheduler
