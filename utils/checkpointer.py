import os
import torch

class Checkpointer(object):
    def __init__(self, model, optimizer=None, scheduler=None, 
            save_dir="", checkpoint_period=10000, compressed_rate=100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.checkpoint_period = checkpoint_period
        self.compressed_rate = compressed_rate
        self.best_acc = 0

    def save(self, name):
        data = {}
        data["model"] = self.model.state_dict()
        data["compressed_rate"] = self.compressed_rate
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        torch.save(data, save_file)

    def save_state_only(self, name):
        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        torch.save(self.model.state_dict(), save_file)
