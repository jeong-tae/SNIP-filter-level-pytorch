import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

def build_augmentation(is_train, param):
    #TODO: implement for custom augmentation
    pass

class Dataloader(data.Dataset):
    def __init__(self, root, split = 'train'):
        #TODO: implement for custom data loader
        pass


def CIFAR10_loader(args, is_train = False):
    if is_train:
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
    dataset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=is_train,
            download=True, transform = transformer)

    # Note: pin_memory should be false when you deploy model which running with small dataset
    dataloader = data.DataLoader(dataset, batch_size = args.batch_size,
            shuffle=is_train, num_workers=args.num_workers, pin_memory=True)

    return dataloader
