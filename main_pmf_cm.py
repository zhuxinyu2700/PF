import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal, xavier_uniform
from torch.distributions import Categorical
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import random
import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from tqdm import tqdm
tqdm.monitor_interval = 0
from utils import *
from preprocess_movie_lens import *
from transD_movielens import *
import joblib
from collections import Counter, OrderedDict
sys.path.append('../')
import gc
from model import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_tqdm', type=bool, default=True, help='')
    parser.add_argument('--save_dir', type=str, default='./results/PMF/orig', help="output path")
    parser.add_argument('--do_log', default=True, help="whether to log to csv")
    parser.add_argument('--load_transD', action='store_true', help="Load TransD")
    parser.add_argument('--load_filters', action='store_true', help="Load TransD")
    parser.add_argument('--freeze_transD', action='store_true', help="Load TransD")
    parser.add_argument('--test_new_disc', default=True, help="Load TransD")
    parser.add_argument('--use_cross_entropy', default=True, help="DemPar Discriminators Loss as CE")
    parser.add_argument('--api_key', type=str, default=" ", help="Api key for Comet ml")
    parser.add_argument('--project_name', type=str, default=" ", help="Comet project_name")
    parser.add_argument('--workspace', type=str, default=" ", help="Comet Workspace")
    parser.add_argument('--D_steps', type=int, default=10, help='Number of D steps')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs (default: 500)')
    parser.add_argument('--num_classifier_epochs', type=int, default=10, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (default: 512)')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Batch size (default: 512)')
    parser.add_argument('--gamma', type=int, default=10, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--valid_freq', type=int, default=1, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency in epochs (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=30, help='Embedding dimension (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--p', type=int, default=2, help='P value for p-norm')
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--use_1M', type=bool, default=True, help='Use 1M dataset')
    parser.add_argument('--use_attr', type=bool, default=True, help='Initialize all Attribute')
    parser.add_argument('--use_occ_attr', type=bool, default=False, help='Use Only Occ Attribute')
    parser.add_argument('--use_gender_attr', type=bool, default=False, help='Use Only Gender Attribute')
    parser.add_argument('--use_age_attr', type=bool, default=False, help='Use Only Age Attribute')
    parser.add_argument('--use_pmf', type=bool, default=True, help='Use a PMF')
    parser.add_argument('--dont_train', action='store_true', help='Dont Do Train Loop')
    parser.add_argument('--sample_mask', type=bool, default=True, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--use_trained_filters', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--optim_mode', type=str, default='adam', help='optimizer')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    args.train_ratings,args.test_ratings,args.users,args.movies = make_dataset_1M(True)

    args.num_users = int(np.max(args.users['user_id'])) + 1
    args.num_movies = int(np.max(args.movies['movie_id'])) + 1
    users = np.asarray(list(set(args.users['user_id'])))
    np.random.shuffle(users)
    cutoff_constant = 0.8
    train_cutoff_row = int(np.round(len(users)*cutoff_constant))
    args.cutoff_row = train_cutoff_row
    args.users_train = users[:train_cutoff_row]
    args.users_test = users[train_cutoff_row:]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.outname_base = os.path.join(args.save_dir,args.namestr+'_MovieLens_results')
    args.saved_path = os.path.join(args.save_dir,args.namestr+'_MovieLens_resultsD_final.pts')
    args.gender_filter_saved_path = args.outname_base + 'GenderFilter.pts'
    args.occupation_filter_saved_path = args.outname_base + 'OccupationFilter.pts'
    args.age_filter_saved_path = args.outname_base + 'AgeFilter.pts'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##############################################################
    return args

def main(args):
    train_set = KBDataset(args.train_ratings, args.prefetch_to_gpu)
    test_set = KBDataset(args.test_ratings, args.prefetch_to_gpu)
    train_fairness_set = NodeClassification(args.users_train, args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(args.users_test, args.prefetch_to_gpu)

    ''' Comet Logging '''
    if args.use_pmf:
        modelD = PMF(args.num_users, args.num_movies, args.embed_dim, args.p).to(args.device)

    ''' Initialize Everything to None '''
    fairD_gender, fairD_occupation, fairD_age, fairD_random = None,None,None,None
    optimizer_fairD_gender, optimizer_fairD_occupation, \
            optimizer_fairD_age, optimizer_fairD_random = None,None,None,None
    gender_filter, occupation_filter, age_filter = None, None, None
    if args.use_attr:
        attr_data = [args.users,args.movies]
        ''' Initialize Discriminators '''
        fairD_gender = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'gender',use_cross_entropy=args.use_cross_entropy).to(args.device)
        fairD_occupation = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_age = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)

        ''' Initialize Optimizers '''
        if args.sample_mask:
            gender_filter = AttributeFilter(args.embed_dim,attribute='gender').to(args.device)
            occupation_filter = AttributeFilter(args.embed_dim,attribute='occupation').to(args.device)
            age_filter = AttributeFilter(args.embed_dim,attribute='age').to(args.device)
            optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
            optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
            optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)

    elif args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_occupation = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
    elif args.use_gender_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'gender',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
    elif args.use_age_attr:
        attr_data = [args.users,args.movies]
        fairD_age = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)

    if args.load_transD:
        modelD.load(args.saved_path)

    if args.load_filters:
        gender_filter.load(args.gender_filter_saved_path)
        occupation_filter.load(args.occupation_filter_saved_path)
        age_filter.load(args.age_filter_saved_path)

    ''' Create Sets '''
    fairD_set = [fairD_gender,fairD_occupation,fairD_age]
    filter_set = [gender_filter,occupation_filter,age_filter]
    optimizer_fairD_set = [optimizer_fairD_gender, optimizer_fairD_occupation, optimizer_fairD_age]

    ''' Initialize CUDA if Available '''
    if args.use_cuda:
        for fairD,filter_ in zip(fairD_set,filter_set):
            if fairD is not None:
                fairD.to(args.device)
            if filter_ is not None:
                filter_.to(args.device)


    if args.sample_mask and not args.use_trained_filters:
        optimizerD = optimizer(list(modelD.parameters()) + \
                list(gender_filter.parameters()) + \
                list(occupation_filter.parameters()) + \
                list(age_filter.parameters()), 'adam', args.lr)
    else:
         optimizerD = optimizer(modelD.parameters(), 'adam', args.lr)


    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)

    if args.freeze_transD:
        freeze_model(modelD)


    ''' Joint Training '''
    if not args.dont_train:
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            if epoch % args.valid_freq == 0 or epoch == 1:
                with torch.no_grad():
                    if args.use_pmf:
                        test_loss, test_hit, test_ndcg = test_pmf(test_set,args,modelD,filter_set)

                print("epoch %d : test PMF Loss is %f " % (epoch, test_loss))
                print("epoch %d : test H@5 is %f " % (epoch, test_hit))
                print("epoch %d : test N@5 is %f " % (epoch, test_ndcg))

            train_pmf(train_loader, epoch, args, modelD, optimizerD, \
                  fairD_set, optimizer_fairD_set, filter_set)
            gc.collect()

        modelD.save(args.outname_base + 'D_final.pts')
        if args.use_attr or args.use_gender_attr:
            fairD_gender.save(args.outname_base + 'GenderFairD_final.pts')
        if args.use_attr or args.use_occ_attr:
            fairD_occupation.save(args.outname_base + 'OccupationFairD_final.pts')
        if args.use_attr or args.use_age_attr:
            fairD_age.save(args.outname_base + 'AgeFairD_final.pts')

        if args.sample_mask:
            gender_filter.save(args.outname_base + 'GenderFilter.pts')
            occupation_filter.save(args.outname_base + 'OccupationFilter.pts')
            age_filter.save(args.outname_base + 'AgeFilter.pts')


        if args.test_new_disc:
            args.use_attr = True
            ''' Training Fresh Discriminators'''
            args.freeze_transD = True
            attr_data = [args.users, args.movies]
            freeze_model(modelD)
            ''' Train Classifier '''
            train_gender(args, modelD, train_fairness_set, test_fairness_set,attr_data, filter_set)
            train_occupation(args, modelD, train_fairness_set, test_fairness_set,attr_data, filter_set)
            train_age(args, modelD, train_fairness_set, test_fairness_set,attr_data, filter_set)
            '''Test Fresh Discriminatots'''
            test_gender(args, test_fairness_set, modelD, fairD_gender, epoch, filter_set)
            test_occupation(args, test_fairness_set, modelD, fairD_occupation, epoch, filter_set)
            test_age(args, test_fairness_set, modelD, fairD_age, epoch, filter_set)


if __name__ == '__main__':
    main(parse_args())
