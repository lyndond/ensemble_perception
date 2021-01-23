"""
This script is called from the shell and trains the models via Slurm arrays
"""

import argparse
import os
import sys
import time
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from data_preprocess import get_train_validation_data
from tqdm import tqdm

matplotlib.use('Agg')


def model_trainer(train_loader: DataLoader, valid_loader: DataLoader, args):
    """"""

    train_logs = {}
    train_logs['accuracy'] = []
    train_logs['loss'] = []
    valid_logs = {}
    valid_logs['accuracy'] = []
    valid_logs['loss'] = []

    # define model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 10)  # replace fully connected

    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        model = model.cuda()

    print('number of parameters is ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    """DEFINE OPTIMIZER"""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    criterion = torch.nn.CrossEntropyLoss()

    """ TRAINING LOOP: LOOP OVER EPOCHS"""
    pbar = tqdm(range(args.num_epochs), desc='Epoch')
    for h in pbar:
        pbar.set_description(f'Epoch {h:d}')
        total_n_correct = 0
        loss_sum = 0

        if h == 0:
            start_time_total = time.time()

        if h >= 30 and h % 30 == 0:
            for param_group in optimizer.param_groups:
                args.lr = args.lr / 2
                param_group["lr"] = args.lr

        # loop over images
        for i, batch in enumerate(train_loader, 0):

            inputs, labels = batch['image'], batch['label']

            model.train()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            n_correct = (outputs.argmax(dim=1) == labels).sum()
            total_n_correct += n_correct.item()

            loss_sum += loss.item()

        # compute avg train loss and accuracy after each epoch
        train_logs['accuracy'].append(total_n_correct / len(train_loader.dataset))
        train_logs['loss'].append(loss_sum / (i + 1))

        if h == 0:
            print("--- %s seconds for one epoch ---" % (time.time() - start_time_total))

        """VALIDATION STEP"""
        model.eval()

        total_n_correct = 0
        loss_sum = 0

        for i, valid_batch in enumerate(valid_loader, 0):
            inputs, labels = valid_batch['image'], valid_batch['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            n_correct = (outputs.argmax(dim=1) == labels).sum()
            total_n_correct += n_correct.item()

            loss_sum += loss.item()

        # Average valid loss and accuracy after each epoch
        valid_logs['accuracy'].append(total_n_correct / len(valid_loader.dataset))
        valid_logs['loss'].append(loss_sum / (i + 1))

        pbar.set_postfix({'val acc':  f"{valid_logs['accuracy'][-1]:.2f}",
                          'val loss': f"{valid_logs['loss'][-1]:.2f}"})

        """ SAVE AND PLOT """
        torch.save(model.state_dict(), args.dir_name + '/model.pt')
        np.save(args.dir_name + '/train_logs.npy', train_logs)
        np.save(args.dir_name + '/valid_logs.npy', valid_logs)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(range(len(train_logs['loss'])), train_logs['loss'], 'b-o', label='train loss')
        axs[0].plot(range(len(valid_logs['loss'])), valid_logs['loss'], 'r-o', label='validation loss')

        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].set_title('min validation loss ' + str(round(min(valid_logs['loss']), 2)) + ' from epoch ' + str(
            valid_logs['loss'].index(min(valid_logs['loss']))))
        axs[0].legend()

        axs[1].plot(range(len(train_logs['accuracy'])), train_logs['accuracy'], 'b-*', label='train accuracy')
        axs[1].plot(range(len(valid_logs['accuracy'])), valid_logs['accuracy'], 'r-*', label='validation accuracy')
        axs[1].set_title('max validation accuracy ' + str(round(max(valid_logs['accuracy']), 2)) + ' from epoch ' + str(
            valid_logs['accuracy'].index(max(valid_logs['accuracy']))))
        axs[1].set_ylabel('accuracy')
        axs[1].set_xlabel('epoch')
        axs[1].legend()

        fig.savefig(args.dir_name + '/loss.png')

    return model


def main():
    """Calls model_trainer() with arguments defined in shell"""

    parser = argparse.ArgumentParser(description='training multiple models')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--dir_name', default='../models/resnet_num_')
    parser.add_argument('--SLURM_ARRAY_TASK_ID', type=int)
    parser.add_argument('--momentum', default=.9)

    args = parser.parse_args()

    """ LOAD THE DATA """
    train_loader, valid_loader = get_train_validation_data(root_dir='./data/imagenette2-320',
                                                           batch_size=int( args.batch_size/2),
                                                           crop_size=256)

    print('number of train images ', len(train_loader.dataset))
    print('number of valid images ', len(valid_loader.dataset))

    # train multiple models and store them in separate folders
    args.dir_name = args.dir_name + str(args.SLURM_ARRAY_TASK_ID)

    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

    model_trainer(train_loader, valid_loader, args)


if __name__ == "__main__":
    main()
