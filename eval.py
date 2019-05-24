import sys
import os.path
import os
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import model2

import utils
import ipdb
import time
import pickle
import random

import numpy as np


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(net, loader):
    """ Run an epoch over the given loader """
    answ = []
    idxs = []
    accs = []
    ss_vc = []

    softmax = nn.Softmax(dim=1).cuda()    ### dim=1    ## to check; np.sum(a_m.tolist())
    for v, q, a, idx, q_len in tqdm(loader):  # image, ques to vocab mapped , answer, item (sth to help index shuffled data with), len_val

        var_params = {
            'volatile': False,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)  ### len of question

        with torch.no_grad():
            out = net(v, q, q_len)

            softmax_vc = softmax(out)   # torch.size(128,3000)
            acc = utils.batch_accuracy(out.data, a.data).cpu()   #torch.Size([128, 1])
        ###print(acc)  ## see what accuracy it prints  - prints the official vqa acc for every questions
        # store information about evaluation of this minibatch
        _, answer = out.data.cpu().max(dim=1)              ### torch.Size([128)  !!!! this is the predicted answer id!!!
        answ.append(answer) #.view(-1))   # pred_ans_id
        ss_vc.append(softmax_vc)       # #torch.Size([128, 3000])
        accs.append(acc) #.view(-1))      # official vqa accurcay per question
        idxs.append(idx) #view(-1).clone())   ###     data.py- line 152 we return `item` so that the order of (v, q, a) triples can be restored if desired
                                                    # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
    ss_vc = list(torch.cat(ss_vc, dim=0))      ## softmax_vectors
    answ = list(torch.cat(answ, dim=0))          ## pred_ans_id
    accs = list(torch.cat(accs, dim=0))          ## official vqa accurcay per question
    idxs = list(torch.cat(idxs, dim=0))             ## some item

    print('the accuracy is:', torch.mean(torch.stack(accs)) )       ### mean of entire accuracy vector # tensor(0.6015) for val set
    return answ, ss_vc, accs, idxs


def main():
    start_time = time.time()

    cudnn.benchmark = True
    output_qids_answers = []
    val_loader = data.get_loader(val=True)            ## data shuffled only in train, so in order here
    ipdb.set_trace()  ##  val_loader.VQA.__getitem__(116591)[0]  print the largest coco_id - this is within torch int64 bound!
    #test_loader = data.get_loader(test=True)


    net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
    model_path = os.path.join(config.model_path_show_ask_attend_answer)
    results_file = os.path.join(config.results_with_attn_pth)
    res_pkl = os.path.join(config.results_with_attn_pkl)

    print('loading model from', model_path)
    net.load_state_dict(torch.load(model_path)["weights"])
    print(net)
    net.eval()
    r = run(net, val_loader)
    results = {
            'ans_id': r[0],
            'official_acc': r[2],
            'idx': r[3],
            'ss_vc': r[1]
        }
    print('run complete')
    torch.save(results, results_file)
    print('saving results to '+ results_file )

    # converting them to python objects:   ##### .item(); .tolist()
    output_qids_answers += [
        { 'ans_id': p.item(), 'ss_vc': np.float32(softmax_vector.tolist())}## int(qid); softmax_vector.tolist()  because json does not recognize NumPy data types. Convert the number to a Python int before serializing the object:
        for p, softmax_vector in zip(r[0],r[1])]                  ### ques-id in order as json files

    with open(res_pkl, 'wb') as f:
        pickle.dump(output_qids_answers,f, pickle.HIGHEST_PROTOCOL)
    print('saving pkl complete')


    print('time_taken:', time.time() - start_time)

if __name__ == '__main__':
    main()
