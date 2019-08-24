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
random.seed(1234)
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', required=True, type=str)  ## 'with_attn' , 'no_attn'
parser.add_argument('--split', required=True, type=str)  ### train2014/val2014/test2015/
parser.add_argument('--edit_set', required=True, type=int)  ## 1 0

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(net, loader, edit_set_cmd):
    """ Run an epoch over the given loader """
    answ = []
    accs = []
    ss_vc = []
    image_ids =[]
    ques_ids = []
    softmax = nn.Softmax(dim=1).cuda()
    for v, q, a, idx, img_id, ques_id, q_len in tqdm(loader):  # image, ques to vocab mapped , answer, item (sth to help index shuffled data with), len_val
        #ipdb.set_trace()
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
            #ipdb.set_trace() ## check type of softmax_vc- enforce it to torch16 here itself/ alse see what happens when np.16..
            acc = utils.batch_accuracy(out.data, a.data).cpu()   #torch.Size([128, 1]) official vqa acc for every questions

        # store information about evaluation of this minibatch
        _, answer = out.data.cpu().max(dim=1)              ### torch.Size([128)  !!!! this is the predicted answer id!!!
        answ.append(answer.view(-1))   # pred_ans_id
        ss_vc.append(softmax_vc)       # #torch.Size([128, 3000])
        accs.append(acc.view(-1))      # official vqa accurcay per question
        ques_ids.append(ques_id.view(-1))

        if edit_set_cmd:
            image_ids.append(img_id)
        else:
            image_ids.append(img_id.view(-1))
            #ipdb.set_trace()
    ss_vc = torch.cat(ss_vc, dim=0)    ## softmax_vectors
    answ = torch.cat(answ, dim=0)       ## pred_ans_id
    accs = torch.cat(accs, dim=0) ## official vqa accurcay per question
    ques_ids = torch.cat(ques_ids, dim=0)
    if edit_set_cmd:
        image_ids = [item for sublist in image_ids for item in sublist]
    else:
        image_ids = torch.cat(image_ids, dim=0)
     ### might be string in edit config case
    print('the accuracy is:', torch.mean(accs))       ### mean of entire accuracy vector # tensor(0.6015) for val set

    return answ, image_ids, ques_ids, ss_vc


def main(args):
    start_time = time.time()
    if args.edit_set:
        print('evaluating on edited VQA')
    else:
        print('evaluating original VQA')

    cudnn.benchmark = True
    output_qids_answers = []

    if args.split == 'train2014':
        _, val_loader = data.get_loader(train=True)    #val=True)            ## data shuffled only in train
    elif args.split == 'val2014':
        _, val_loader = data.get_loader(val=True)
    elif args.split == 'test2015':
        _, val_loader = data.get_loader(test=True)
    #test_loader = data.get_loader(test=True)

    if args.model_type == 'no_attn':
        net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_no_attn)
        results_file = os.path.join(config.results_no_attn_pth)
        res_pkl = os.path.join(config.results_no_attn_pkl)

    elif args.model_type == 'with_attn':
        net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_show_ask_attend_answer)
        results_file = os.path.join(config.results_with_attn_pth)
        res_pkl = os.path.join(config.results_with_attn_pkl)

    print('loading model from', model_path)
    net.load_state_dict(torch.load(model_path)["weights"])   ### so here you load the weights, essentially the model
    print(net)
    net.eval()
    r = run(net, val_loader, args.edit_set)

    print('saving results to '+ res_pkl )

    if args.edit_set:
        output_qids_answers += [
            { 'ans_id': p.item(), 'img_id': id,'ques_id':qid.item(),'ss_vc': np.float16(softmax_vector.tolist())}  #np.float32(softmax_vector).tolist()
            for p,id,qid, softmax_vector in zip(r[0],r[1], r[2], r[3])]
    else:
        output_qids_answers += [
            { 'ans_id': p.item(), 'img_id': id.item(),'ques_id':qid.item(),'ss_vc': np.float16(softmax_vector.tolist())}
            for p,id,qid, softmax_vector in zip(r[0],r[1], r[2], r[3])]
    with open(res_pkl, 'wb') as f:
        pickle.dump(output_qids_answers,f, pickle.HIGHEST_PROTOCOL)
    print('saving pkl complete')


    print('time_taken:', time.time() - start_time)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
