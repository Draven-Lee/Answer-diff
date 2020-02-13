from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets
import time
import copy
import json
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from focalloss import FocalLoss, compute_accuracy, CBLoss
from dataset import AnswerDisDataset
from model import init_model

plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=2, suppress=True)


def evaluate_model(model, dataloader):
    model.eval()
    score_all = []
    label_all = []

    # Iterate over data.
    for images, questions, answers, labels in dataloader:
        images = images.to(device)
        questions = questions.to(device)
        labels = labels.to(device)
        answers = answers.to(device)
        outputs = model(images, questions, answers)
        #outputs = torch.sigmoid(outputs)

        score_all.append(outputs.data.cpu().numpy())
        label_all.append(labels.data.cpu().numpy())

    score_all = np.concatenate(score_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    ap = average_precision_score(label_all, score_all, average=None)
    ap[np.isnan(ap)] = 0.

    return ap, label_all, score_all


def train_model(model, num_epochs, train_splits=['train'],
                eval_splits=['test'], n_epochs_per_eval=1):
    criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss()
    params = [{'params': model.base_model.parameters()}]
    optimizer = optim.Adam(params, lr=1e-3)
    # Decay LR by a factor of 0.1 every 100 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_ap = 0.0

    train_dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,
                                                        shuffle=True, num_workers=4) for x in train_splits}

    eval_dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,
                                                       shuffle=False, num_workers=4) for x in eval_splits}

    dataloaders = {}
    dataloaders.update(train_dataloaders)
    dataloaders.update(eval_dataloaders)


    ###########evaluate init model###########
    for eval_split in eval_splits:
        ap, answer, score = evaluate_model(model, dataloaders[eval_split])
        print('(AP={1}) {0}'.format(eval_split, 100 * ap))
        ap = np.mean(ap)
        print(100 * ap)
    print()
    #########################################

    running_loss = 0.0
    i = 0

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_corrects = 0

        # Iterate over data.
        for train_split in train_splits:
            for images, questions, answers, labels in dataloaders[train_split]:
                model.train()  # Set model to training mode
                images = images.to(device)
                questions = questions.to(device)
                labels = labels.to(device)
                answers = answers.to(device)
                outputs = model(images, questions, answers)
                #print(labels.size(), labels[:, 0].size())
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                i += 1

                if i % 10 == 9:
                    print('loss_total = {:.4f}'.format(running_loss/10, i))
                    writer.add_scalar('training loss', running_loss/10, i)
                    running_loss = 0.0

        scheduler.step()

        # compute average precision
        if (epoch + 1) % n_epochs_per_eval == 0:
            for eval_split in eval_splits:
                ap, label, score = evaluate_model(model, dataloaders[eval_split])
                print('(AP={1}) {0}'.format(eval_split, 100 * ap))
                ap = np.mean(ap)
                print(100 * ap)
                #name = 'label' + str(epoch) + '.npy'
                #np.save(name, label)
                #name = 'score' + str(epoch) + '.npy'
                #np.save(name, score)
            # deep copy the model

            if ap > best_ap:
                best_ap = ap
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    ###########evaluate final model###########
    for eval_split in eval_splits:
        ap, label, score = evaluate_model(model, dataloaders[eval_split])
        print('(AP={1}) {0}'.format(eval_split, 100 * ap))
        ap = np.mean(ap)
        print(100 * ap)
    # deep copy the model
    if ap > best_ap:
        best_ap = ap
        best_model_wts = copy.deepcopy(model.state_dict())
    print()
    #########################################

    print('Best val AP: {:2f}'.format(100 * best_ap))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return


if __name__ == '__main__':
    writer = SummaryWriter('runs/vqa_experiment')
    splits = ['train', 'val', 'test']
    datasets = {}
    datasets.update({x: AnswerDisDataset('vqa', x) for x in splits})
    #datasets.update({x: AnswerDisDataset('vizwiz', x) for x in splits})
    dataset_sizes = {x: len(datasets[x]) for x in splits}
    print(dataset_sizes)

    train_splits = ['train', 'val']
    model_type_list = ['Q+I+A']
    eval_splits = ['test']

    for model_type in model_type_list:
        print(model_type)
        model = init_model(datasets, model_type)
        save_model_path = './models/{0}_train-on-({1}).pt'.format(model_type, ','.join(train_splits))
        train_model(model, num_epochs=5, train_splits=train_splits, eval_splits=eval_splits, n_epochs_per_eval=1)
        torch.save(model.state_dict(), save_model_path)
        print('\n')
