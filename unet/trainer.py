import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.functional import cross_entropy_loss_with_logits


class Trainer:
    grad_com_size = 1
    def __init__(
            self,
            model: torch.nn.Module,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            device="cpu"):
        self.dataloader = dataloader
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.progress = None
        self.counter = 0

    def _train_instance(self, image, label):
        output = self.model(image)
        loss = 0
        loss = cross_entropy_loss_with_logits(output, label, 1.1)
        self.counter = self.counter + 1
        loss = loss / self.grad_com_size
        loss.backward()
        if self.counter == self.grad_com_size:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.counter = 0
        self.progress.update(1)

    def train(self):
        print("\ntrain start\n")
        self.model.train()
        self.optimizer.zero_grad()

        self.counter = 0
        self.progress = tqdm(total=len(self.dataloader))
        for i, (image, label) in enumerate(self.dataloader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            self._train_instance(image, label)
        self.progress.clear()
        self.progress = None