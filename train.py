import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
from model import *
import argparse


ori_img_path = "/home/lsm/PycharmProjects/Unet_ISBI_pytorch/raw/train"
ori_list = sorted(os.listdir(ori_img_path))
# print(ori_list)
train_img_list, val_img_list = ori_list[:24], ori_list[24:]



label_img_path = "/home/lsm/PycharmProjects/Unet_ISBI_pytorch/raw/label"
label_list = sorted(os.listdir(label_img_path))
# print(label_list)
train_label_list, val_label_list = label_list[:24], label_list[24:]

class UnetDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, img_root=ori_img_path, label_root=label_img_path,
                 img_list=None, label_list=None,
                 transform=None, target_transform=None):
        assert img_root is not None and label_root is not None, 'Must specify img_root and label_root!'
        self.img_root = img_root
        self.label_root = label_root
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = Image.open(os.path.join(self.label_root, self.label_list[index]))
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label
        else:
            return image


    def __len__(self):
        return len(self.img_list)


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

parser = argparse.ArgumentParser(description='PyTorch Unet ISBI Challenge')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
parser.add_argument('--log_interval', type=int, default=3, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--save_model_name', default='my_unet.pth',
                        help='name of saved model')
parser.add_argument('--save_folder', default='checkpoints/',
                        help='Directory for saving checkpoint models')
args = parser.parse_args()
if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class toBinary(object):
    def __call__(self, label):
        label = np.array(label)
        # print(image)
        label = label * (label > 127)
        label = Image.fromarray(label)
        return label

transform_image = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
])

transform_label = transforms.Compose([
    transforms.Grayscale(),
    toBinary(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4938, 0.4933, 0.4880), (0.1707, 0.1704, 0.1672)),
])

train_dataset = UnetDataset(img_list=train_img_list, label_list=train_label_list,
                            transform=transform_image, target_transform=transform_label)
val_dataset = UnetDataset(img_list=val_img_list, label_list=val_label_list,
                          transform=transform_image, target_transform=transform_label)

kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = UNet(1, 1)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()
sig = nn.Sigmoid()
best_dice = 0.0


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    dice = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print("data shape:{}".format(data.dtype))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) # This loss is per image
        sig_output = sig(output)
        dice += dice_coeff(sig_output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    dice_acc = 100. * dice / len(train_loader)
    print('Train Dice coefficient: {:.2f}%'.format(dice_acc))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    dice = 0
    global best_dice
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            sig_output = sig(output)
            dice += dice_coeff(sig_output, target)


    test_loss /= len(test_loader.dataset)
    dice_acc = 100. * dice / len(test_loader)

    print('\nTest set: Batch average loss: {:.4f}, Dice Coefficient: {:.2f}%\n'.format(test_loss, dice_acc))

    if dice_acc > best_dice:
        torch.save(model.state_dict(), args.save_folder + args.save_model_name)
        best_dice = dice_acc
        print("======Saving model======")

def main():
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, val_loader)

if __name__ == '__main__':
    main()