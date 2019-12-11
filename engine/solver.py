import torch

def build_optimizer(args, model):
    # Not support other optimizer yet
    print(" [*] optimizer is SGD")
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer

def build_lr_scheduler(args, optimizer):
    
    if 'multistep' in args.scheduler_type.lower():
        print(" [*] lr_scheduler is multistepLR")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decaysteps, gamma=args.lr_anneal)
    else:
        raise NotImplementedError(" [!] Not implemented scheduler yet")

    return lr_scheduler
