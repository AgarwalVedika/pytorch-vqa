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
parser.add_argument('--model_type', default= 'with_attn', type=str)  ## 'with_attn' , 'no_attn' ,  'finetuning_SAAA'



def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(net, loader, edit_set_cmd):
    """ Run an epoch over the given loader """

    accs = []

    answ = []
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
            if config.vis_attention:
                out =
            else:
                out = net(v, q, q_len)
            softmax_vc = softmax(out)   # torch.size(128,3000)
            #ipdb.set_trace() ## check type of softmax_vc- enforce it to torch16 here itself/ alse see what happens when np.16..
            acc = utils.batch_accuracy(out.data, a.data).cpu()   #torch.Size([128, 1]) official vqa acc for every questions

        # store information about evaluation of this minibatch
        _, answer = out.data.cpu().max(dim=1)              ### torch.Size([128)  !!!! this is the predicted answer id!!!



        ipdb.set_trace()
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

    cudnn.benchmark = True
    _, val_loader = data.get_loader(val=True)
    #test_loader = data.get_loader(test=True)

    if args.model_type == 'no_attn':
        net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_no_attn)
        res_pkl = os.path.join(config.results_no_attn_pkl)

    elif args.model_type == 'with_attn':
        net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_show_ask_attend_answer)
        res_pkl = os.path.join(config.results_with_attn_pkl)

    elif args.model_type == 'finetuning_CNN_LSTM':
        net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
        print()
        print('testing on ', config.test_data_split)
        print()
        model_trained_data_splits = ['orig_10', 'orig_all', 'orig_10_edit_10','orig_all_edit_10',  'orig_all_edit_all']
        for model_trained_data_split in model_trained_data_splits:
            model_path_folder = os.path.join('./models/' + config.model_type + '/' + config.ques_type + '/' + model_trained_data_split)
            model_path = os.path.join(model_path_folder, 'epoch_{}.pth'.format(str(config.ft_val_10_naive_CNN_LSTM_edit_more15[config.ques_type.replace(' ', '_') + '_' + model_trained_data_split])))
            res_json_folder = os.path.join(config.ft_logs_folder)
            os.makedirs(res_json_folder, exist_ok=True)
            res_json = os.path.join(res_json_folder, 'results_fineuned_using_' + model_trained_data_split + '.json')

            print('loading model from', model_path)
            net.load_state_dict(torch.load(model_path)["weights"])   ### so here you load the weights, essentially the model
            #print(net)
            net.eval()
            accuracy = run(net, val_loader)
            dumping_dict ={}
            dumping_dict['model_path'] = model_path
            dumping_dict['accuracy'] = float(accuracy)

            print(config.ques_type ,model_trained_data_split)
            print('accuracy on ',config.test_data_split, ': ', accuracy )

            with open(res_json, 'w') as f:
                json.dump(dumping_dict, f)

    elif args.model_type == 'finetuning_SAAA':
        net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
        print()
        print('testing on ', config.test_data_split)
        print()
        model_trained_data_splits = ['orig_10', 'orig_all', 'orig_10_edit_10','orig_all_edit_10',  'orig_all_edit_all']
        for model_trained_data_split in model_trained_data_splits:
            model_path_folder = os.path.join('./models/' + config.model_type + '/' + config.ques_type + '/' + model_trained_data_split)
            model_path = os.path.join(model_path_folder, 'epoch_{}.pth'.format(str(config.ft_val_10_naive_SAAA[config.ques_type.replace(' ', '_') + '_' + model_trained_data_split])))


            res_json_folder = os.path.join(config.ft_logs_folder)
            os.makedirs(res_json_folder, exist_ok=True)
            res_json = os.path.join(res_json_folder, 'results_fineuned_using_' + model_trained_data_split + '.json')

            print('loading model from', model_path)
            net.load_state_dict(torch.load(model_path)["weights"])   ### so here you load the weights, essentially the model
            #print(net)
            net.eval()
            accuracy = run(net, val_loader)
            dumping_dict ={}
            dumping_dict['model_path'] = model_path
            dumping_dict['accuracy'] = float(accuracy)

            print(config.ques_type ,model_trained_data_split)
            print('accuracy on ',config.test_data_split, ': ', accuracy )

            with open(res_json, 'w') as f:
                json.dump(dumping_dict, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
