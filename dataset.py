import os
import json
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import nltk
import torch
import numpy as np
from PIL import Image


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
        #label = [label[0]]
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
