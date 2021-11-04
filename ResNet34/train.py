import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from tqdm import tqdm

from model import resnet34
from my_dataset import MyDataSet
from data_utils import read_split_data, plot_class_preds
from train_eval_utils import train_one_epoch, evaluate, train_acc, val_loss

from torch.utils.tensorboard import SummaryWriter

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    tb_writer = SummaryWriter(log_dir="runs/tumor_experiment")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    random.seed(0)
    images_path = "./data/Train/"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(images_path)

    batch_size = 16
    nw = 0

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model = resnet34(num_classes=4).to(device)
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(model, init_img)

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # fully connected layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)
    epochs = 50

    for epoch in range(epochs):
        # train
        training_loss = train_one_epoch(model=net,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device)
        print("[epoch {}] mean training loss: {}".format(epoch, round(training_loss, 3)))

        training_accuracy = train_acc(model=net,
                                      data_loader=train_loader,
                                      device=device)
        print("[epoch {}] training accuracy: {}".format(epoch, round(training_accuracy, 3)))

        # validate
        validation_accuracy = evaluate(model=net,
                       data_loader=validate_loader,
                       device=device)
        print("[epoch {}] mean validation accuracy: {}".format(epoch, round(validation_accuracy, 3)))
        validation_loss = val_loss(model=net,
                                   data_loader=validate_loader,
                                   device=device)

        print("[epoch {}] validation loss: {}".format(epoch, round(validation_loss, 3)))

        tags = ["training_loss", "validation_loss", "training_accuracy", "validation_accuracy"]
        tb_writer.add_scalar(tags[0], training_loss, epoch)
        tb_writer.add_scalar(tags[1], validation_loss, epoch)
        tb_writer.add_scalar(tags[2], training_accuracy, epoch)
        tb_writer.add_scalar(tags[3], validation_accuracy, epoch)
        #optimizer.param_groups[0]["lr"]

        # add figure into tensorboard
        fig = plot_class_preds(net=net,
                               images_dir="./data/Test/glioma_tumor/image(1).jpg",
                               transform=data_transform["val"],
                               num_plot=5,
                               device=device)
        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch)

        # save weights
        torch.save(net.state_dict(), "./weights/1020-1818-16-0.001-model-{}.pth".format(epoch))

    print('Finished Training')


if __name__ == '__main__':
    main()
