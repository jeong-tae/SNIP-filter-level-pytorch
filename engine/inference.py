from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
sys.path.append("..")
from utils import ClassWiseAverageMeter, AverageMeter, accuracy

def inference(model, data_loader, logger, device, current_epoch, args):
    model.eval()

    classwise_acc = ClassWiseAverageMeter(args.num_classes)
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    val_loss = AverageMeter()

    for images, labels in tqdm(data_loader, desc="Inference"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)
            loss = F.cross_entropy(output, labels)
            acc_ith = accuracy(output, labels, topk=(1, 2, 3))

            acc1.update(acc_ith[0].item(), labels.size(0))
            acc2.update(acc_ith[1].item(), labels.size(0))
            acc3.update(acc_ith[2].item(), labels.size(0))
            val_loss.update(loss.item(), labels.size(0))
            classwise_acc.update(output, labels)

    logger.scalar_summary("val_acc1", acc1.avg, current_epoch)
    logger.scalar_summary("val_acc2", acc2.avg, current_epoch)
    logger.scalar_summary("val_acc3", acc3.avg, current_epoch)
    logger.scalar_summary("val_loss", val_loss.avg, current_epoch)
    classwise_acc.get_average()
    logger.scalar_summary("val_clswise_acc", classwise_acc.top1_avg.mean(), current_epoch)
    print(" [*] val_acc1: %.4f"%acc1.avg)
    return acc1.avg, classwise_acc.top1_avg.mean()


