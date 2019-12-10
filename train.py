import argparse
import os, time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils import Logger, Checkpointer, ClassWiseAverageMeter, AverageMeter
from engine import build_optimizer, build_lr_scheduler, inference
from models import build_model, SNIP
from data import CIFAR10_loader
import shutil

def model_compression(model, data_loader, device, kappa = 0.9):
    compressor = SNIP(model, device, kappa)
    input_x, input_y = next(iter(data_loader))
    input_x = input_x.to(device)
    input_y = input_y.to(device)

    compressed_model, comp_rate = compressor.construct_small_network(input_x, input_y)
    return compressed_model, comp_rate

def run_train(args):
    device = torch.device(args.device)

    model = build_model(args.model_name, args.num_classes, args.pretrained)
    model = model.to(device)

    train_loader = CIFAR10_loader(args, is_train = True)
    test_loader = CIFAR10_loader(args, is_train = False)

    model, comp_rate = model_compression(model, train_loader, device, kappa = args.kappa)

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = Checkpointer(model, optimizer, scheduler, args.experiment, args.checkpoint_period, comp_rate)
    logger = Logger(os.path.join(args.experiment, 'tf_log'))

    acc1, _ = inference(model, test_loader, logger, device, 0, args)
    checkpointer.best_acc = acc1
    for epoch in tqdm(range(0, args.max_epoch)):
        train_epoch(model, train_loader, optimizer, len(train_loader)*epoch, checkpointer, device, logger)
        acc1, m_acc1 = inference(model, test_loader, logger, device, epoch+1, args)
        if acc1 > checkpointer.best_acc:
            checkpointer.save("model_best")
            checkpointer.best_acc = acc1
        scheduler.step()
    
    checkpointer.save("model_last")

def train_epoch(model, train_loader, optimizer, current_iter, checkpointer, device, logger):
    model.train()

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        t1 = time.time()
        output = model(images)
        loss = F.cross_entropy(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        if current_iter % 20 == 0:
            print(" [*] Iter %d || Loss: %.4f || Timer: %.4f"%(current_iter, loss.item(), t2 - t1))

        if current_iter % checkpointer.checkpoint_period == 0:
            checkpointer.save("model_{:07}".format(current_iter))

        logger.scalar_summary("train_loss", loss.item(), current_iter)
        current_iter += 1

    return current_iter

def main():
    parser = argparse.ArgumentParser(description = "Training arguments for KD")
    parser.add_argument('--lr', default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument('--batch-size', default=128, help="# of dataset you forward at once")
    parser.add_argument('--num-workers', default=4, help="# of worker to queue your dataset")
    parser.add_argument('--momentum', default=0.9, help="Rate to accumulates the gradient of the past steps")
    parser.add_argument('--max-epoch', default=30, type=int, help="Maximum epoch to train")
    parser.add_argument('--lr-decaysteps', default=[15, 25], nargs='+', type=int, help="Decay the learning rate at given steps")
    parser.add_argument('--lr-anneal', default=0.1, help="Multiplicative factor of lr decay")
    parser.add_argument('--kappa', default=0.9, help="Target sparsity level for filters")
    parser.add_argument('--weight-decay', default=5e-4, help="L2 regularization coefficient")
    parser.add_argument('--scheduler-type', default='multistep', help="")
    parser.add_argument('--model_name', default='resnet101', help="Model name to train")
    parser.add_argument('--device', default='cuda', type=str, help="To enable GPU. set 'cuda', otherwise 'cpu'")
    parser.add_argument('--pretrained', default=False, action='store_true', help="Set:True starts from imagenet pretrained model")
    parser.add_argument('--num_classes', default=10, type=int, help="# of classes in the dataset")
    parser.add_argument('--experiment', default='cifar10_snip_resnet101_1', type=str, help="Path to save your experiment")
    parser.add_argument('--checkpoint-period', default=5000, type=int, help="Frequency of model save based on # of iterations")
    args = parser.parse_args()
    args.experiment = 'experiment/' + args.experiment

    if os.path.exists(args.experiment):
        print(" [*] %s already exists. Process may overwrite existing experiemnt"%args.experiment)
    else:
        print(" [*] New experiment is set. Create directory at %s"%args.experiment)
        os.makedirs(args.experiment)
    shutil.copy2('./train_snip.sh', args.experiment+'/train_snip.sh')

    run_train(args)
    print(" [*] Done! Results are saved in %s"%args.experiment)

if __name__ == "__main__":
    main()


