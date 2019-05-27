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
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default= 'with_attn', type=str)  ## 'with_attn' , 'no_attn'
parser.add_argument('--edit_set', required=True, type=bool)  ## True False

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
    softmax = nn.Softmax(dim=1).cuda()    ### dim=1    ## to check; np.sum(a_m.tolist())
    for v, q, a, idx, img_id, ques_id, q_len in tqdm(loader):  # image, ques to vocab mapped , answer, item (sth to help index shuffled data with), len_val

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

        # answ.append(answer) #.view(-1))   # pred_ans_id
        # ss_vc.append(softmax_vc)       # #torch.Size([128, 3000])
        # accs.append(acc) #.view(-1))      # official vqa accurcay per question
        # image_ids.append(img_id)
        # ques_ids.append(ques_id)

        # second perhaps faster way
        answ.append(answer.view(-1))   # pred_ans_id
        ss_vc.append(softmax_vc)       # #torch.Size([128, 3000])
        accs.append(acc.view(-1))      # official vqa accurcay per question
        ques_ids.append(ques_id.view(-1))

        if edit_set_cmd:
            image_ids.append(img_id)
        else:
            image_ids.append(img_id.view(-1))


    # ss_vc = [item.tolist() for sublist in ss_vc for item in sublist]#torch.cat(ss_vc, dim=0).tolist()    ## softmax_vector
    # answ = [item.item() for sublist in answ for item in sublist]   #torch.cat(answ, dim=0).tolist()     ## pred_ans_id
    # accs = [item.item() for sublist in accs for item in sublist]#torch.cat(accs, dim=0).tolist()           ## official vqa accurcay per question
    # ques_ids = [item.item() for sublist in ques_ids for item in sublist]#torch.cat(ques_ids, dim=0).tolist()
    # image_ids=[item for sublist in image_ids for item in sublist]   ### might be string in edit config case
    # print('accuracy is', mean(accs))

    ## second perhaps faster way
    ss_vc = list(torch.cat(ss_vc, dim=0))      ## softmax_vectors
    answ = list(torch.cat(answ, dim=0))          ## pred_ans_id
    accs = list(torch.cat(accs, dim=0))          ## official vqa accurcay per question
    ques_ids = list(torch.cat(ques_ids, dim=0))
    if edit_set_cmd:
        image_ids = [item for sublist in image_ids for item in sublist]
    else:
        image_ids = list(torch.cat(image_ids, dim=0))
     ### might be string in edit config case
    print('the accuracy is:', torch.mean(torch.stack(accs)) )       ### mean of entire accuracy vector # tensor(0.6015) for val set

    return answ, image_ids, ques_ids, ss_vc


def main(args):
    start_time = time.time()

    cudnn.benchmark = True
    output_qids_answers = []
    val_loader = data.get_loader(val=True)            ## data shuffled only in train, so in order here
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
    net.load_state_dict(torch.load(model_path)["weights"])
    print(net)
    net.eval()
    r = run(net, val_loader, args.edit_set)

    print('saving results to '+ res_pkl )

    # converting them to python objects:   ##### .item(); .tolist()
    # output_qids_answers += [
    #     { 'ans_id': p, 'img_id': id,'ques_id':qid,'ss_vc': np.float32(softmax_vector)}## int(qid); softmax_vector.tolist()  because json does not recognize NumPy data types. Convert the number to a Python int before serializing the object:
    #     for p,id,qid, softmax_vector in zip(r[0],r[1], r[2], r[3])]                  ### ques-id in order as json files
    # second perhaps faster way
    if args.edit_set:
        output_qids_answers += [
            { 'ans_id': p.item(), 'img_id': id,'ques_id':qid.item(),'ss_vc': np.float32(softmax_vector.tolist())}
            for p,id,qid, softmax_vector in zip(r[0],r[1], r[2], r[3])]
    else:
        output_qids_answers += [
            { 'ans_id': p.item(), 'img_id': id.item(),'ques_id':qid.item(),'ss_vc': np.float32(softmax_vector.tolist())}
            for p,id,qid, softmax_vector in zip(r[0],r[1], r[2], r[3])]
    with open(res_pkl, 'wb') as f:
        pickle.dump(output_qids_answers,f, pickle.HIGHEST_PROTOCOL)
    print('saving pkl complete')


    print('time_taken:', time.time() - start_time)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
