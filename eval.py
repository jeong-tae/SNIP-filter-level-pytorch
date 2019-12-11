import time, argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F

from data import CIFAR10_loader

def run_eval(args):
    device = torch.device(args.device)

    # load model, you can't use state_dict to load with original network shape.
    ckpt = torch.load(args.model_path)
    model = ckpt['model']
    model.eval()

    test_loader = CIFAR10_loader(args, is_train = False)
    
    test_loss = 0
    correct = 0
    total = 0
    times = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device) 

            t0 = time.time()
            output = model(images)
            t1 = time.time()
            times += (t1-t0)
            loss = F.cross_entropy(output, labels)
            test_loss += loss.item()
            total += labels.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(labels).sum().item()

    fps = 1. / (times/total)
    acc = 100.*correct/total
    print("Acc: %.4f%%, FPS: %.2f, comp_rate: %.2f"%(acc, fps, ckpt['compressed_rate']))

def main():
    parser = argparse.ArgumentParser(description = "Evaluation arguments for SNIP")
    parser.add_argument('--batch-size', default=128, type=int, help="# of dataset you forward at once")
    parser.add_argument('--num-workers', default=4, help="# of worker to queue your dataset")
    parser.add_argument('--device', default='cuda', type=str, help="To enable GPU. set 'cuda', otherwise 'cpu'")
    parser.add_argument('--model_path', default='', type=str, help="Model path that you trained the model")
    args = parser.parse_args()

    run_eval(args)

if __name__ == "__main__":
    main()
