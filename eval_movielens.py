from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import numpy as np
import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
from utils import *
from user_item import *
sys.path.append('../../')
from model import *

user_dict = load_dict("user_dict.json")

def optimizer(params, mode, *args, **kwargs):
    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True,
                weight_decay=1e-4, **kwargs)
    else:
        raise NotImplementedError()
    return opt

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
    y_test = np.asarray(y_test).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_pred, average=average)

def test_random(args,test_dataset,modelD,net,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, probs_list, labels_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
        correct = correct + preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = roc_auc_score(cat_labels_list,cat_probs_list,average="micro")
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list,cat_preds_list,average='binary')
    print("Test Random Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))

def test_gender(args,test_dataset,modelD,net,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, probs_list, labels_list  = [], [], []
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
        correct = correct + preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = roc_auc_score(cat_labels_list,cat_probs_list,average="micro")
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list,cat_preds_list,average='binary')
    print("Test Gender Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))


def train_gender(args,modelD,train_dataset,test_dataset,attr_data,filter_set=None):
    modelD.eval()
    net = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'gender',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.BCELoss()

    for epoch in range(1,args.num_classifier_epochs + 1):
        correct = 0
        embs_list, labels_list = [], []
        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = roc_auc_score(y.data.cpu().numpy(),\
                    y_hat.data.cpu().numpy(),average="micro")
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(),\
                    average='binary')
            if epoch == args.num_classifier_epochs:
                embs_list.append(p_batch_emb)
                labels_list.append(y)
            print("Train Gender Loss is %f Accuracy is: %f AUC: %f F1:%f"\
                    %(loss,acc,AUC,f1))
        if epoch % 10 == 0:
            test_gender(args,test_dataset,modelD,net,epoch,filter_set)


def test_age(args,test_dataset,modelD,net,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, labels_list, probs_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
        correct = correct + preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = multiclass_roc_auc_score(cat_labels_list,cat_probs_list)
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list, cat_preds_list, average='micro')
    print("Test Age Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))

def train_age(args,modelD,train_dataset,test_dataset,attr_data,\
        filter_set=None):
    modelD.eval()
    net = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'age',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.NLLLoss()

    for epoch in range(1,args.num_classifier_epochs+1):
        correct = 0
        embs_list, labels_list = [], []
        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = multiclass_roc_auc_score(y.data.cpu().numpy(),y_hat.data.cpu().numpy())
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(), average='micro')
            if epoch == args.num_classifier_epochs:
                embs_list.append(p_batch_emb)
                labels_list.append(y)
            print("Train Age Loss is %f Accuracy is: %f AUC: %f F1: %f" \
                    %(loss,acc,AUC,f1))
        if epoch % 10 == 0:
            test_age(args,test_dataset,modelD,net,epoch,filter_set)


def test_occupation(args,test_dataset,modelD,net,epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=8000)
    correct = 0
    preds_list, labels_list, probs_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
        correct = correct + preds.eq(y.view_as(preds)).sum().item()
        probs_list.append(y_hat)
        preds_list.append(preds)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    try:
        f1 = f1_score(cat_labels_list, cat_preds_list, average='micro')
        AUC = multiclass_roc_auc_score(cat_labels_list,cat_probs_list)
        acc = 100. * correct / len(test_dataset)
        print("Test Occupation Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))

    except:
        acc = 100. * correct / len(test_dataset)


def train_occupation(args,modelD,train_dataset,test_dataset,\
        attr_data,filter_set=None):
    modelD.eval()
    net = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'occupation',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=8000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.NLLLoss()

    for epoch in range(1,args.num_classifier_epochs+1):
        correct = 0
        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = multiclass_roc_auc_score(y.data.cpu().numpy(),y_hat.data.cpu().numpy())
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(), average='micro')
            if epoch == args.num_classifier_epochs:
                embs_list.append(p_batch_emb)
                labels_list.append(y)
            print("Train Occupation Loss is %f Accuracy is: %f AUC: %f F1: %f"\
                    %(loss,acc,AUC,f1))
        if epoch % 10 == 0:
            test_occupation(args, test_dataset, modelD, net, epoch, filter_set)
        embs_list, labels_list = [], []


def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def test_pmf(dataset, args, modelD,filter_set=None):
    test_loader = DataLoader(dataset, batch_size=4000, num_workers=1, collate_fn=collate_fn)
    data_itr = enumerate(test_loader)

    test_loss_list = []
    hit_list = []
    ndcg_list = []
    for idx, p_batch in data_itr:
        p_batch_var = Variable(p_batch).cuda()
        test_loss, hit, ndcg = modelD(p_batch_var,return_nh=True,filters=filter_set)
        test_loss_list.append(test_loss)
        hit = torch.tensor(hit)
        ndcg = torch.tensor(ndcg)
        hit_list.append(hit)
        ndcg_list.append(ndcg)
    test_loss = torch.mean(torch.stack(test_loss_list))
    test_hit = torch.mean(torch.stack(hit_list))
    test_ndcg = torch.mean(torch.stack(ndcg_list))
    return test_loss, test_hit, test_ndcg

def validate_pmf(dataset, args, modelD,filter_set=None):
    test_loader = DataLoader(dataset, batch_size=4000, num_workers=1, collate_fn=collate_fn)
    data_itr = enumerate(test_loader)

    test_loss_list = []
    for idx, p_batch in data_itr:
        p_batch_var = Variable(p_batch).cuda()
        test_loss = modelD(p_batch_var,filters=filter_set)
        test_loss_list.append(test_loss)
    test_loss = torch.mean(torch.stack(test_loss_list))
    return test_loss