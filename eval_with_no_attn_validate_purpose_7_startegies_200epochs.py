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
parser.add_argument('--model_type', default= 'finetuning_CNN_LSTM', type=str)  ## 'with_attn' , 'no_attn'
parser.add_argument('--edit_set_cmd', default=1, type=int)  ## 1 0    ### so this stays 1 always as now i have modified the data loader - image_id : orig/edit: both are string!
parser.add_argument('--save_cmd', default=1, type=int)  ## 1 0


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(net, loader, edit_set_cmd, save_cmd):
    """ Run an epoch over the given loader """

    accs = []
    if save_cmd:
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
            out = net(v, q, q_len)
            softmax_vc = softmax(out)   # torch.size(128,3000)
            #ipdb.set_trace() ## check type of softmax_vc- enforce it to torch16 here itself/ alse see what happens when np.16..
            acc = utils.batch_accuracy(out.data, a.data).cpu()   #torch.Size([128, 1]) official vqa acc for every questions

        # store information about evaluation of this minibatch
        _, answer = out.data.cpu().max(dim=1)              ### torch.Size([128)  !!!! this is the predicted answer id!!!

        accs.append(acc.view(-1))  # official vqa accurcay per question
        if save_cmd:
            answ.append(answer.view(-1))  # pred_ans_id
            ss_vc.append(softmax_vc)  # #torch.Size([128, 3000])
            ques_ids.append(ques_id.view(-1))
            if edit_set_cmd:
                image_ids.append(img_id)
            else:
                image_ids.append(img_id.view(-1))


    accs = torch.cat(accs, dim=0)  ## official vqa accurcay per question
    if save_cmd:
        ss_vc = torch.cat(ss_vc, dim=0)  ## softmax_vectors
        answ = torch.cat(answ, dim=0)  ## pred_ans_id
        ques_ids = torch.cat(ques_ids, dim=0)
        if edit_set_cmd:
            image_ids = [item for sublist in image_ids for item in sublist]
        else:
            image_ids = torch.cat(image_ids, dim=0)
        print('the accuracy is:', torch.mean(accs))  ### mean of entire accuracy vector # tensor(0.6015) for val set

        return answ, image_ids, ques_ids, ss_vc
    else:
        return torch.mean(accs)


def main(args):
    start_time = time.time()

    cudnn.benchmark = True
    output_qids_answers = []

    _, val_loader = data.get_loader(val=True)
    #test_loader = data.get_loader(test=True)

    # if args.model_type == 'no_attn':
    #     net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
    #     model_path = os.path.join(config.model_path_no_attn)
    #     res_pkl = os.path.join(config.results_no_attn_pkl)
    #
    # elif args.model_type == 'with_attn':
    #     net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
    #     model_path = os.path.join(config.model_path_show_ask_attend_answer)
    #     res_pkl = os.path.join(config.results_with_attn_pkl)

    if args.model_type == 'finetuning_CNN_LSTM' or args.model_type == 'data_aug_CNN_LSTM':
        net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()

    elif args.model_type == 'finetuning_SAAA' or args.model_type == 'data_aug_SAAA':
        net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()

    print()
    print('testing on ', config.test_data_split)
    print()

    epoch_list = [i for i in range(0, 200, 1)]
    model_types_grand = ['finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0.3_MSE_1', 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0',
                         'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_1', 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0_KL_0.3_MSE_0']

    GIVE_INDEX = 0

    model_types = [model_types_grand[GIVE_INDEX] for i in  range(200)]
    #epoch_list = [i for i in range(0, 200, 1)]
    #model_types = [model_types_grand[GIVE_INDEX] for i in range(200)]

    for idx,model_type in enumerate(model_types):
        if model_type == 'finetuning_CNN_LSTM' or model_type == 'finetuning_SAAA':
            model_trained_data_split = 'orig_10'
        else:
            model_trained_data_split = 'orig_10_edit_10'   # 'orig_all_edit_10'

        if config.ques_type=='0.1_0.0' or config.ques_type=='0.1_0.1':
            model_trained_data_split = 'orig_all_edit_10'

        model_path_folder = os.path.join('./models/' + model_type + '/' + config.ques_type + '/' + model_trained_data_split)
        model_path = os.path.join(model_path_folder, 'epoch_{}.pth'.format(epoch_list[idx]))

        # geting epoch 00 model....results
        # model_path = './models/show_ask_attend_answer.pth'
        # res_json_folder = os.path.join('/BS/vedika3/nobackup/pytorch-vqa/finetuning_logs/EPOCH_00_SAAA/what color is the/entire_logs_using_epoch-1/', config.test_data_split)
        # os.makedirs(res_json_folder, exist_ok=True)
        # res_pkl = os.path.join(res_json_folder, 'results_fineuned_using_no_finetuning_epoch_00_model.pickle')

        res_json_folder = os.path.join('/BS/vedika3/nobackup/pytorch-vqa/finetuning_logs/' +  model_type + '/' + config.ques_type + '/entire_logs_using_epoch' + str(epoch_list[idx]) +  '/' + config.test_data_split)
        os.makedirs(res_json_folder, exist_ok=True)
        res_json = os.path.join(res_json_folder, 'results_fineuned_using_' + model_trained_data_split + '.json')
        res_pkl = os.path.join(res_json_folder, 'results_fineuned_using_' + model_trained_data_split + '.pickle')

        print('loading model from', model_path)
        net.load_state_dict(torch.load(model_path)["weights"])   ### so here you load the weights, essentially the model
        #print(net)
        net.eval()

        if args.save_cmd:
            r = run(net, val_loader, args.edit_set_cmd, args.save_cmd)
            print('saving results to ' + res_pkl)

            if args.edit_set_cmd:
                output_qids_answers += [
                    {'ans_id': p.item(), 'img_id': id, 'ques_id': qid.item(),
                     'ss_vc': np.float16(softmax_vector.tolist())}  # np.float32(softmax_vector).tolist()
                    for p, id, qid, softmax_vector in zip(r[0], r[1], r[2], r[3])]
            else:
                output_qids_answers += [
                    {'ans_id': p.item(), 'img_id': id.item(), 'ques_id': qid.item(),
                     'ss_vc': np.float16(softmax_vector.tolist())}
                    for p, id, qid, softmax_vector in zip(r[0], r[1], r[2], r[3])]
            with open(res_pkl, 'wb') as f:
                pickle.dump(output_qids_answers, f, pickle.HIGHEST_PROTOCOL)
            print('saving pkl complete')
            print('time_taken:', time.time() - start_time)

        else:
            accuracy = run(net, val_loader, args.edit_set_cmd, args.save_cmd)
            dumping_dict ={}
            dumping_dict['model_path'] = model_path
            dumping_dict['accuracy'] = float(accuracy)

            print(config.ques_type ,model_type)
            print('accuracy on ',config.test_data_split, ': ', accuracy )

            with open(res_json, 'w') as f:
                json.dump(dumping_dict, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
