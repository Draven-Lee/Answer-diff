from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
from torch.utils.tensorboard import SummaryWriter
from language_model import WordEmbedding, QuestionEmbedding
from fc import FCNet, FCNetDrop
from torch.nn.utils.weight_norm import weight_norm
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import Dataset
import skimage
from PIL import Image
from copy import deepcopy
from inspect import signature
import time
import pickle
import nltk
from attention import NewAttention, AllAttention
import matplotlib.pyplot as plt
from focalloss import FocalLoss, compute_accuracy

plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=2, suppress=True)


class AnswerDisDataset(Dataset):
    """
    To create the dataset class
    """
    def __init__(self, dataset_name, split):
        super(AnswerDisDataset, self).__init__()

        self.image_dir = None
        if dataset_name == 'vqa':
            self.image_dir = './data/vqa'
        elif dataset_name == 'vizwiz':
            self.image_dir = './data/vizwiz'
        self.image_ext = '.jpg'

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'trainval': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.transform = data_transforms[split]
        self.word2vocab_id = json.load(open('word2vocab_id.json'))
        self.ans2id = json.load(open('ans2id.json'))

        dataroot = './Annotations'
        dataset = json.load(open(os.path.join(dataroot, '%s_ans_difference_%s.json' % (dataset_name, split))))
        max_length = 20
        for sample in dataset:
            question = sample['question']
            question = question.lower()
            tokens = nltk.word_tokenize(question)
            tokens = [self.word2vocab_id[x] for x in tokens if x in self.word2vocab_id]
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [0] * (max_length - len(tokens))
                tokens = padding + tokens
            sample['q_token'] = tokens

            tokens = [x.lower() for x in sample['answers']]
            tokens = [self.ans2id[x] for x in tokens if x in self.ans2id]
            tokens = tokens[:10]
            if len(tokens) < 10:
                # Note here we pad in front of the sentence
                padding = [0] * (10 - len(tokens))
                tokens = padding + tokens
            sample['a_token'] = tokens
        self.dataset = dataset

    def __getitem__(self, index):
        entry = self.dataset[index]
        image = entry['image']
        image_path = os.path.join(self.image_dir, image.replace('.jpg', self.image_ext))
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        label = [0 if x < 2 else 1 for x in entry['ans_dis_labels']]
        label = torch.tensor(label, dtype=torch.float32)

        question = torch.from_numpy(np.array(entry['q_token']))
        #answer = np.zeros((len(self.ans2id)), dtype=np.float32)
        #for ans in entry['a_token']:
        #    answer[ans] = 1.0
        ##answer /= 10.0
        ## the score for correct answer and 0 for others, score is between (0,1)
        #answer = torch.tensor(answer, dtype=torch.float32)
        answer = torch.from_numpy(np.array(entry['a_token']))

        if self.transform:
            image = self.transform(image)

        return image, question, answer, label

    def __len__(self):
        return len(self.dataset)


class Identity(nn.Module):
    def __init__(self, dim):
        super(Identity, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x


def init_image_model(init_model_path=None):
    """
    use Resnet50 as pretrained model
    """
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.avgpool = Identity(num_ftrs)
    model.fc = Identity(num_ftrs)

    if init_model_path:
        model.load_state_dict(torch.load(init_model_path))
    model = model.to(device)
    return model


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, a_net, classifier, model_type='Q+I'):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.a_net = a_net
        self.model_type = model_type
        self.num_hid = 1024
        self.a_emb = WordEmbedding(len(datasets['train'].ans2id), 300, 0.0)
        self.a_emb.init_embedding('glove6b_init_300d_answer.npy')##self.a_emb = [batch, 10, 300]
        self.aq_emb = QuestionEmbedding(300, self.num_hid, 1, False, 0.0) #qq_emb = [batch, 1024]
        self.av_att = AllAttention(2048, self.num_hid, self.num_hid)
        self.i_net = FCNet([2048, self.num_hid])


        self.classifier = classifier



    def forward(self, v, q, a):
        """Forward

        return: logits, not probs

        v represents the image features[batch, -1, 2048]
        q represents the question features[batch, 20]
        a [batch, 6250], w_emb: [batch, 20, 300], q_emb: [32, 1024]
        att: [32, 49, 1], v_emb: [32, 2048], q_repr: [32, 1024]
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        a_emb = self.a_emb(a)
        aq_emb = self.aq_emb(a_emb)

        a_att = self.av_att(v, q_emb, aq_emb)
        va_emb = (a_att*v).sum(1)

        va_emb = self.i_net(va_emb)


        q_repr = self.q_net(q_emb)
        va_repr = self.v_net(v.sum(1))
        a_repr = self.a_net(aq_emb)

        #print(va_emb.size(), q_repr.size(), a_repr.size(),va_repr.size())

        joints = (va_emb + q_repr + va_repr + a_repr)/4
        logits = self.classifier(joints)


        return logits


class FullModel(nn.Module):
    def __init__(self, image_model, base_model):
        super(FullModel, self).__init__()
        self.image_model = image_model
        self.base_model = base_model
        #self.drop = nn.Dropout(0.5)


    def forward(self, img, q, a):
        x = self.image_model(img)
        x = x.view(x.size(0), -1, self.image_model.fc.dim)
        #x = self.drop(x)
        x = self.base_model(x, q, a)
        return x


def init_model(model_type='Q+I'):
    num_hid = 1024
    image_model = init_image_model()
    v_dim = image_model.fc.dim
    w_emb = WordEmbedding(len(datasets['train'].word2vocab_id), 300, 0.0)
    w_emb.init_embedding('glove6b_init_300d.npy')
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([image_model.fc.dim, num_hid])
    a_net = FCNet([num_hid, num_hid])
    classifier = weight_norm(nn.Linear(num_hid, 10), dim=None)

    base_model = BaseModel(w_emb, q_emb, v_att, q_net, v_net, a_net, classifier, model_type)
    base_model = base_model.to(device)
    model = FullModel(image_model, base_model)
    return model

def evaluate_model(model, dataloader):
    model.eval()
    score_all = []
    label_all = []
    #answer_all = []

    # Iterate over data.
    for images, questions, answers, labels in dataloader:
        images = images.to(device)
        questions = questions.to(device)
        labels = labels.to(device)
        answers = answers.to(device)
        outputs = model(images, questions, answers)

        #print(outputs.size())

        score_all.append(outputs.data.cpu().numpy())
        label_all.append(labels.data.cpu().numpy())
        #answer_all.append(answers.data.cpu().numpy())

    score_all = np.concatenate(score_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    #print(np.shape(score_all), np.shape(label_all))
    #score_all[np.isnan(score_all)] = 0.
    #answer_all[np.isnan(answer_all)] = 0.
    ap = average_precision_score(label_all, score_all, average=None)
    #ap[np.isnan(ap)] = 0.

    return ap, label_all, score_all


def train_model(model, num_epochs=5, train_splits=['train'],
                eval_splits=['test'], n_epochs_per_eval=1):
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()
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
            # deep copy the model

            if ap > best_ap:
                best_ap = ap
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print(flush=True)

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
    dataset_sizes = {x: len(datasets[x]) for x in splits}
    print(dataset_sizes)

    train_splits = ['train', 'val']
    model_type_list = ['Q+I+A']
    eval_splits = ['test']

    for model_type in model_type_list:
        print(model_type)

        model = init_model(model_type)
        save_model_path = './models/{0}_train-on-({1}).pt'.format(model_type, ','.join(train_splits))
        train_model(model, num_epochs=5, train_splits=train_splits, eval_splits=eval_splits, n_epochs_per_eval=1)
        torch.save(model.state_dict(), save_model_path)
        print('\n')