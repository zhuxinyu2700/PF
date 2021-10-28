import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import pickle
import json
import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import Counter
from numpy.random import RandomState
from user_item import *
from utils import *
sys.path.append('../../')
import heapq

''' Some Helpful Globals '''
ltensor = torch.LongTensor
user_dict = load_dict("user_dict.json")

def apply_filters(emb, masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_embed = 0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_embed += filter_(emb)
    return filter_embed

def dcg_at_k(r, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.array(r)
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k( sorted(r, reverse=True), method )   # IDCG
    if not dcg_max:
        return 0.
    return dcg_at_k(r, method) / dcg_max

def hit_at_k(r):
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


class PMF(nn.Module):
    def __init__(self, num_users, num_items, enbed_dim, p):
        super(PMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.enbed_dim = enbed_dim
        self.random_state = RandomState(1)
        self.users_embed = nn.Embedding(num_users, enbed_dim, sparse=False)
        self.users_embed.weight.data = torch.from_numpy(0.1 * self.random_state.rand(num_users, enbed_dim)).float()
        self.items_embed = nn.Embedding(num_items, enbed_dim, sparse=False)
        self.items_embed.weight.data = torch.from_numpy(0.1 * self.random_state.rand(num_items, enbed_dim)).float()
        self.criterion = nn.LogSigmoid()

    def encode(self, users, items, filters=None):
        users_embed = self.users_embed(users)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant != 0:
                users_embed = apply_filters(users_embed, filters)
        items_embed = self.items_embed(items)
        return users_embed, items_embed

    def forward(self, batch, return_embed=False,return_nh=False, filters = None):
        users, items = batch[:,0], batch[:,2]
        neg_items = []
        users_embed, items_embed = self.encode(users, items, filters)
        for u in users:
            neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
            while neg_item in user_dict[str(u.item())]:
                neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
            else:
                neg_items.append(neg_item)
        neg_items = torch.LongTensor(neg_items)
        neg_items_embed = self.items_embed(neg_items)
        R_neg = (users_embed * neg_items_embed).sum(1)
        R_pos = (users_embed * items_embed).sum(1)
        loss = -self.criterion(R_pos-R_neg).mean()
        if not return_embed and not return_nh:
            return loss
        elif not return_embed and return_nh:
            test_list = batch.cpu().numpy().tolist()
            hit = 0
            ndcg = 0
            r = []
            for i in range(len(test_list)):
                test_items = np.random.randint(low=0, high=self.num_items, size=100)
                test_items = test_items.tolist()
                j = 0
                while j < len(test_items):
                    if test_items[j] in user_dict[str(test_list[i][0])]:
                        del test_items[j]
                    else:
                        j += 1
                test_items.append(test_list[i][2])
                test_items_t = torch.LongTensor(test_items)
                u = torch.tensor(test_list[i][0])
                user_embed, test_items_embed = self.encode(u, test_items_t, filters)
                R = (user_embed * test_items_embed).sum(1)
                score = R.cpu().numpy().tolist()
                item_score = dict(zip(test_items, score))
                item_score = sorted(item_score.items(), key=lambda item: item[1], reverse=True)
                item_score = dict(item_score[:5])
                if str(test_list[i][2]) in item_score:
                    r.append(1)
                else:
                    r.append(0)
            hit += hit_at_k(r)
            ndcg += ndcg_at_k(r,method=1)
            return loss, hit/len(test_list), ndcg/len(test_list)
        else:
            return loss, users_embed, items_embed

    def get_embed(self, users, filters=None):
        with torch.no_grad():
            users_embed = self.users_embed(users)
            if filters is not None:
                constant = len(filters) - filters.count(None)
            if constant != 0:
                users_embed = apply_filters(users_embed, filters)
        return users_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))


class GenderDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(GenderDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_sex = attribute_data[0]['sex']
        users_sex = [0 if i == 'M' else 1 for i in users_sex]
        self.users_sensitive = np.ascontiguousarray(users_sex)
        self.out_dim = 1
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2 ), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = self.sigmoid(scores)
        A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.sigmoid(scores)
            A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class AgeDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(AgeDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        self.criterion = nn.NLLLoss()
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_age = attribute_data[0]['age'].values
        users_age_list = sorted(set(users_age))
        if not use_1M:
            bins = np.linspace(5, 75, num=15, endpoint=True)
            inds = np.digitize(users_age, bins) - 1
            self.users_sensitive = np.ascontiguousarray(inds)
            self.out_dim = len(bins)
        else:
            reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
            inds = [reindex.get(n, n) for n in users_age]
            self.users_sensitive = np.ascontiguousarray(inds)
            self.out_dim = 7

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = output.max(1, keepdim=True)[1] # get the index of the max
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class OccupationDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(OccupationDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        self.criterion = nn.NLLLoss()
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_occupation = attribute_data[0]['occupation']
        if use_1M:
            self.users_sensitive = np.ascontiguousarray(users_occupation.values)
            self.out_dim = 21
        else:
            users_occupation_list = sorted(set(users_occupation))
            occ_to_idx = {}
            for i, occ in enumerate(users_occupation_list):
                occ_to_idx[occ] = i
            users_occupation = [occ_to_idx[occ] for occ in users_occupation]
            self.users_sensitive = np.ascontiguousarray(users_occupation)
            self.out_dim = len(users_occupation_list)

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = output.max(1, keepdim=True)[1] # get the index of the max
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class AttributeFilter(nn.Module):
    def __init__(self, embed_dim, attribute='gender'):
        super(AttributeFilter, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute = attribute
        self.W1 = nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True)
        self.W2 = nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True)
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)

    def forward(self, ents_emb):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h2 = self.batchnorm(h2)
        return h2

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


