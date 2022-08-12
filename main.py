import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import utils
from train import per_epoch_activity
from module import GoogLeNet
from datetime import datetime

# 1. define some parameters and create datasets

BATCH_SIZE = 16
EPOCH = 16
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = utils.get_device()
root = r"A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\LeNet_project\data"
model_path = r"A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\GoogLeNet_Project\saved_model\full_model" \
             r"\GoogLeNet_01.pth "

train_aug, val_aug = utils.image_augmentation()
train_set, val_set = utils.get_FashionMNIST(train_aug, val_aug, root=root)
train_loader, val_loader = utils.creat_dataloader(train_set, val_set, BATCH_SIZE)
summary_writer = SummaryWriter()

# 2.
model = GoogLeNet()
model.to(device)
# 3.
loss_fn = nn.CrossEntropyLoss()

# 4.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    per_epoch_activity(train_loader, val_loader, device, optimizer, model, loss_fn, summary_writer, EPOCH, timestamp)
    torch.save(model, model_path)
