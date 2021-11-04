import sys

from tqdm import tqdm
import torch

#training
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader,desc='training (calculating loss) ...')
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()

#validation
def evaluate(model, data_loader, device):
    model.eval()

    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
   
    data_loader = tqdm(data_loader, desc="calculating validation accuracy...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
        
    acc = sum_num.item() / num_samples

    return acc

#calculate training accuracy
def train_acc(model, data_loader, device):
    model.eval()

    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)

    data_loader = tqdm(data_loader, desc="calculating training accuracy...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    acc = sum_num.item() / num_samples

    return acc

#calculate validation loss
def val_loss(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, desc='calculating validation loss...')
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

    return mean_loss.item()
