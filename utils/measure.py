import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ClassWiseAverageMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.num_classes)
        self.top1 = torch.zeros(self.num_classes)
        self.top2 = torch.zeros(self.num_classes)
        self.top3 = torch.zeros(self.num_classes)
        self.top1_avg = torch.zeros(self.num_classes)
        self.top2_avg = torch.zeros(self.num_classes)
        self.top3_avg = torch.zeros(self.num_classes)
        self.top1_avg_all = torch.zeros(self.num_classes)
        self.top2_avg_all = torch.zeros(self.num_classes)
        self.top3_avg_all = torch.zeros(self.num_classes)

    def update(self, output, target, topk=(1, 2, 3), type=None):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        if len(target.size()) > 1:
            target = (target != 0).nonzero()[:, 1]

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for i in range(batch_size):
            self.count[target[i]] = self.count[target[i]] + 1
            self.top1[target[i]] = self.top1[target[i]] + int(correct[:1].sum(0)[i])
            self.top2[target[i]] = self.top2[target[i]] + int(correct[:2].sum(0)[i])
            self.top3[target[i]] = self.top3[target[i]] + int(correct[:3].sum(0)[i])

    def get_average(self):
        self.top1_avg = (self.top1 / self.count)
        self.top2_avg = (self.top2 / self.count)
        self.top3_avg = (self.top3 / self.count)

        self.top1_avg_all = self.top1_avg.mean()
        self.top2_avg_all = self.top2_avg.mean()
        self.top3_avg_all = self.top3_avg.mean()

def accuracy(output, target, topk=(1,), type=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    if len(target.size()) > 1:
        target = (target!=0).nonzero()[:,1]
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
