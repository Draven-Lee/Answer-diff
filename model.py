import torch
import torch.nn as nn
from torchvision import datasets, models
from language_model import WordEmbedding, QuestionEmbedding
from fc import FCNet
from torch.nn.utils.weight_norm import weight_norm
from attention import NewAttention, AllAttention
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=2, suppress=True)


class Identity(nn.Module):
    def __init(self, dim):
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
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, a_net, classifier, data, model_type='Q+I'):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.a_net = a_net
        self.model_type = model_type
        self.num_hid = 1024
        self.a_emb = WordEmbedding(len(data['train'].ans2id), 300, 0.0)
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

        #joints = torch.cat((va_emb, q_repr, a_repr, va_repr), 1)
        joints = (va_emb+va_repr+q_repr+a_repr)/4
        logits = self.classifier(joints)
        return logits


class FullModel(nn.Module):
    def __init__(self, image_model, base_model):
        super(FullModel, self).__init__()
        self.image_model = image_model
        self.base_model = base_model

    def forward(self, img, q, a):
        x = self.image_model(img)
        x = x.view(x.size(0), -1, self.image_model.fc.dim)
        x = self.base_model(x, q, a)
        return x


def init_model(database, model_type='Q+I'):
    num_hid = 1024
    image_model = init_image_model()
    v_dim = image_model.fc.dim
    w_emb = WordEmbedding(len(database['train'].word2vocab_id), 300, 0.0)
    w_emb.init_embedding('glove6b_init_300d.npy')
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([image_model.fc.dim, num_hid])
    a_net = FCNet([num_hid, num_hid])
    classifier = weight_norm(nn.Linear(num_hid, 10), dim=None)
    data = database

    base_model = BaseModel(w_emb, q_emb, v_att, q_net, v_net, a_net, classifier, data, model_type)
    base_model = base_model.to(device)
    model = FullModel(image_model, base_model)
    return model
