import os
import math
import argparse

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from PIL import Image
import torch
from torch.utils.data import Dataset

from vit_model import vit_base as create_model_base_16
from vit_model import vit_large as create_model_large_16
from vit_utils import read_and_split_data, train_oneepoch, evaluate

# Customize dataset
class customize_DataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

# the training main function
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_and_split_data(args.data_path)
    # data processing
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # get the training dataset
    train_dataset = customize_DataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # get the validation dataset
    val_dataset = customize_DataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of dataloader workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # Select model(ViT-Base or ViT-Large ):
    print('\nModel Selection:\n1. Base_16\n2. Large_16\n')
    selection = input('Enter selection: ')
    if selection == '1':
        model = create_model_base_16(num_classes=4, has_logits=True).to(device)
        print('Model \'ViT_base_patch_16_224_in21k\' is selected.\nProceed->>\n\n')
    elif selection == '2':
        model = create_model_large_16(num_classes=4, has_logits=True).to(device)
        print('Model \'ViT_large_patch_16_224_in21k\' is selected.\nProceed->>\n\n')
    else:
        print("Invalid input\nProgram Aborted\n")
        return 0

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # delete unnecessary weights
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    # freeze layer for using pretrained weights
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # freeze weights except head and pre_logits
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer as SGD or Adam(But here we choose SGD)
    optimizer = optim.SGD(pg, lr=args.learningratio, momentum=0.9, weight_decay=5E-5)
    learningf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.learningratiof) + args.learningratiof
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=learningf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_oneepoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["learningratio"], epoch)

        torch.save(model.state_dict(), "./weights/modele-{}.pth".format(epoch))

# set hyperparameters and some model related parameters.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)   # number of classfication classes, we have 4 different classes
    parser.add_argument('--epochs', type=int, default=20)     # num of epochs
    parser.add_argument('--batch_size', type=int, default=32) # batch size
    # learning rate of the optimizer
    parser.add_argument('--learningratio', type=float, default=0.001)
    parser.add_argument('--learningratiof', type=float, default=0.01)

    # the directory of the dataset
    parser.add_argument('--data_path', type=str,
                        default="C:\\Users\\11194\\OneDrive\\Desktop\\archive\\Training")
    parser.add_argument('--model_name', default='', help='create model name')

    # Pre-trained weight path(here we use vit-large pretrained weight)
    parser.add_argument('--weights', type=str,
                        default='C:\\Users\\11194\\OneDrive\\Desktop\\archive\\vit_large_patch16_224_in21k.pth',
                        help='initial weights path')

    # whether freeze the weights
    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    all_args = parser.parse_args()

    main(all_args)