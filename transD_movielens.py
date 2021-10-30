import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
sys.path.append('../../')
from model import *
from eval_movielens import *

ftensor = torch.FloatTensor
ltensor = torch.LongTensor
v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True

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

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def mask_fairDiscriminators(discriminators, mask):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(discriminators, mask) if s)


def train_pmf(data_loader,counter,args,modelD,optimizerD,\
        fairD_set, optimizer_fairD_set, filter_set):
    fairD_gender_loss, fairD_occupation_loss, fairD_age_loss = 0, 0, 0

    if args.show_tqdm:
        data_itr = tqdm(enumerate(data_loader))
    else:
        data_itr = enumerate(data_loader)

    for idx, p_batch in data_itr:
        ''' Sample Fairness Discriminators '''
        if args.sample_mask:
            mask = np.random.choice([0, 1], size=(3,))
            masked_fairD_set = list(mask_fairDiscriminators(fairD_set, mask))
            masked_optimizer_fairD_set = list(mask_fairDiscriminators(optimizer_fairD_set, mask))
            masked_filter_set = list(mask_fairDiscriminators(filter_set, mask))
        else:
            ''' No mask applied despite the name '''
            masked_fairD_set = fairD_set
            masked_optimizer_fairD_set = optimizer_fairD_set
            masked_filter_set = filter_set

        if args.use_cuda:
            p_batch = p_batch.cuda()

        p_batch_var = Variable(p_batch)

        ''' Number of Active Discriminators '''
        constant = len(masked_fairD_set) - masked_fairD_set.count(None)

        ''' Update GCMC Model '''
        if constant != 0:
            task_loss, lhs_emb, rhs_emb = modelD(p_batch_var,return_embed=True, filters=masked_filter_set)
            filter_l_emb = lhs_emb[:len(p_batch_var)]
            l_penalty = 0

            # ''' Apply Filter or Not to Embeddings '''
            # filter_l_emb = apply_filters_gcmc(args,p_lhs_emb,masked_filter_set)

            ''' Apply Discriminators '''
            for fairD_disc, fair_optim in zip(masked_fairD_set, masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    l_penalty = l_penalty + fairD_disc(filter_l_emb, p_batch[:, 0], True)

            if not args.use_cross_entropy:
                fair_penalty = constant - l_penalty
            else:
                fair_penalty = -1 * l_penalty

            if not args.freeze_transD:
                optimizerD.zero_grad()
                full_loss = task_loss + args.gamma * fair_penalty
                full_loss.backward(retain_graph=False)
                optimizerD.step()

            for k in range(0, args.D_steps):
                l_penalty_2 = 0
                for fairD_disc, fair_optim in zip(masked_fairD_set, masked_optimizer_fairD_set):
                    if fairD_disc is not None and fair_optim is not None:
                        fair_optim.zero_grad()
                        l_penalty_2 = l_penalty_2 + fairD_disc(filter_l_emb.detach(), p_batch[:, 0], True)
                        if not args.use_cross_entropy:
                            fairD_loss = -1 * (1 - l_penalty_2)
                        else:
                            fairD_loss = l_penalty_2
                        fairD_loss.backward(retain_graph=True)
                        fair_optim.step()
        else:
            task_loss= modelD(p_batch_var)
            fair_penalty = Variable(torch.zeros(1)).cuda()
            optimizerD.zero_grad()
            full_loss = task_loss + args.gamma * fair_penalty
            full_loss.backward(retain_graph=False)
            optimizerD.step()

        if constant != 0:
            gender_correct, occupation_correct, age_correct, random_correct = 0, 0, 0, 0
            correct = 0
            for fairD_disc in masked_fairD_set:
                if fairD_disc is not None:
                    ''' No Gradients Past Here '''
                    with torch.no_grad():
                        task_loss, lhs_emb, rhs_emb = modelD(p_batch_var, \
                                                                    return_embed=True, filters=masked_filter_set)
                        p_lhs_emb = lhs_emb[:len(p_batch)]
                        filter_emb = p_lhs_emb
                        probs, l_A_labels, l_preds = fairD_disc.predict(filter_emb, p_batch[:, 0], True)
                        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                        if fairD_disc.attribute == 'gender':
                            fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                            gender_correct = gender_correct + l_correct  #
                        elif fairD_disc.attribute == 'occupation':
                            fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                            occupation_correct = occupation_correct+l_correct
                        elif fairD_disc.attribute == 'age':
                            fairD_age_loss = fairD_loss.detach().cpu().numpy()
                            age_correct = age_correct + l_correct

        print("train PMF Loss is %f " % (task_loss))

    ''' Logging for end of epoch '''
    if not args.freeze_transD:
        print(": Task Loss", float(full_loss))
    if fairD_set[0] is not None:
        print(": Fair Gender Disc Loss", float(fairD_gender_loss))
    if fairD_set[1] is not None:
        print("+ Fair Occupation Disc Loss", float(fairD_occupation_loss))
    if fairD_set[2] is not None:
        print(": Fair Age Disc Loss", float(fairD_age_loss))

def train(data_loader, counter, args, modelD, optimizerD,\
         fairD_set, optimizer_fairD_set, filter_set):

    ''' This Function Does Training based on NCE Sampling, for GCMC switch to
    another train function which does not need NCE Sampling'''
    if args.use_pmf:
        train_pmf(data_loader,counter,args,modelD,optimizerD,\
                fairD_set, optimizer_fairD_set, filter_set)

