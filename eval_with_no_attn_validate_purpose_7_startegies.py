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

parser.add_argument('--model_type_no_ft_epoch00', default= 1, type=int)  ## 'with_attn' , 'no_attn'  ### geting epoch 00 model....results

parser.add_argument('--test_training_set', default=0, type=int)  ## 'with_attn' , 'no_attn'  ### geting epoch 00 model....results

parser.add_argument('--edit_set_cmd', default=1, type=int)  ## 1 ONLY   ### so this stays 1 always as now i have modified the data loader - image_id : orig/edit: both are string!
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

    if args.test_training_set:
        print('')
        print('TESTING ON TRAINING SET!!!')
        print('')
        _, val_loader = data.get_loader(train=True)
    else:
        _, val_loader = data.get_loader(val=True)


    # if args.model_type == 'no_attn':
    #     net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
    #     model_path = os.path.join(config.model_path_no_attn)
    #     res_pkl = os.path.join(config.results_no_attn_pkl)
    #
    # elif args.model_type == 'with_attn':
    #     net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
    #     model_path = os.path.join(config.model_path_show_ask_attend_answer)
    #     res_pkl = os.path.join(config.results_with_attn_pkl)


    print()
    print('testing on ', config.test_data_split)
    print()
    #model_trained_data_splits = ['orig_10', 'orig_all', 'orig_10_edit_10','orig_all_edit_10',  'orig_all_edit_all']


    model_types = ['finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0.3_MSE_0',
                          'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0.3_MSE_0',
                          'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0.3_MSE_0',

                          'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0_MSE_1',
                          'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0_MSE_1',
                          'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0_MSE_1',

                          'finetuning_CNN_LSTM_ratio_mix_no_repeat_edit_ids_orig:edit_0.5:0.5',
                          'finetuning_CNN_LSTM_ratio_mix_no_repeat_edit_ids_orig:edit_0.5:0.5',
                          'finetuning_CNN_LSTM_ratio_mix_no_repeat_edit_ids_orig:edit_0.5:0.5',

                          'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.5:0.5',
                          'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.5:0.5',
                          'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.5:0.5',

                          'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.7:0.3',
                          'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.7:0.3',
                          'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.7:0.3']
    epoch_list = [1, 18, 49,      29, 29, 49,      14, 31, 49,       4, 27, 49,      17, 17, 49]


    # model_types_grand = ['finetuning_CNN_LSTM', 'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.7:0.3', 'finetuning_CNN_LSTM_ratio_mix_load_only_orig_ids_no_repeat_edit_ids_orig:edit_0.5:0.5',
    #                      'finetuning_CNN_LSTM_data_aug2_get_edits_origamt_0.5', 'finetuning_CNN_LSTM_data_aug2_get_edits_origamt_0.66',
    #                      'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0.3_KL_0_MSE_1', 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0_KL_0.3_MSE_0',
    #                      'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0.3_KL_0.3_MSE_1']

    model_types_grand = ['finetuning_SAAA_ratio_mix_orig:edit_0.7:0.3',
                         'finetuning_SAAA_data_aug2_get_edits_origamt_0.66',
                         'finetuning_SAAA']

    model_types_grand = ['finetuning_CNN_LSTM', 'finetuning_SAAA']
    model_types_grand = ['data_aug_CNN_LSTM_0.1_0.0_orig_all_edit10' ]

    model_types_grand = ['finetuning_CNN_LSTM', 'finetuning_CNN_LSTM_naive']
    model_types_grand = ['finetuning_SAAA_lr_e-5', 'finetuning_SAAA_lr_e-5_naive']

    model_types_grand = ['finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_CE_0.3_KL_0_MSE_0']

    #model_types_grand = ['finetuning_SAAA_naive', 'finetuning_CNN_LSTM_naive', 'finetuning_CNN_LSTM', 'finetuning_SAAA']


    #model_types_grand = ['finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0.3_MSE_1', 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0',
    #                     'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_1', 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0_KL_0.3_MSE_0']

    model_types_grand = ['finetuning_CNN_LSTM',  'finetuning_CNN_LSTM_naive' , 'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.7:0.3',
                         'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0', ]

    model_types_grand = ['finetuning_CNN_LSTM', 'finetuning_CNN_LSTM_naive']
    model_types_grand = ['finetuning_SAAA', 'finetuning_SAAA_naive']
    model_types_grand = [ 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0', 'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.7:0.3']
    model_types_grand = ['finetuning_SAAA_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0', 'finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0']
    model_types_grand = ['finetuning_CNN_LSTM_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0']
    #model_types_grand = ['finetuning_SAAA_data_aug3_get_edits_origamt_0.66_newCE_0.3_KL_0_MSE_0']
    #model_types_grand = [ 'finetuning_CNN_LSTM', 'finetuning_CNN_LSTM_naive', 'finetuning_SAAA', 'finetuning_SAAA_naive']

    model_types_grand = ['finetuning_CNN_LSTM_naive', 'data_aug_CNN_LSTM_naive']
    model_types_grand = [ 'finetuning_SAAA_naive']
    # ## counting
    # model_types_grand = ['finetuning_CNN_LSTM', 'finetuning_CNN_LSTM_naive_cnt_orig_del', 'finetuning_CNN_LSTM_naive_cnt_orig_edit_del']
    # model_types_grand = ['finetuning_SAAA', 'finetuning_SAAA_naive_cnt_orig_del',  'finetuning_SAAA_naive_cnt_orig_edit_del']
    #model_types_grand = [ 'finetuning_SAAA', 'finetuning_SAAA_naive', ]
    #['finetuning_SAAA_lr_e-5_naive_cnt_orig_del', 'finetuning_SAAA_lr_e-5_naive_cnt_orig_edit_del']
    #model_types_grand = ['finetuning_SAAA_lr_e-7', 'finetuning_SAAA_lr_e-7_naive_cnt_orig_del', 'finetuning_SAAA_lr_e-7_naive_cnt_orig_edit_del']


    GIVE_INDEX = 0

    epoch_list0 = [i for i in range(0, 50, 1)]
    #model_types = [model_types_grand[GIVE_INDEX] for i in range(50)]

    ## if 2 things in model_types grand

    epoch_list = []
    for i in range(len(model_types_grand)):
        epoch_list += epoch_list0

    model_types = []
    for idx in range(len(model_types_grand)):
        model_types +=  [model_types_grand[idx] for i in  range(50)]


    if args.model_type_no_ft_epoch00:
        model_types = ['CNN_LSTM', 'SAAA']


    for idx,model_type in enumerate(model_types):

        if config.ques_type=='0.1_0.0' or config.ques_type=='0.1_0.1':  # model_types_grand = ['finetuning_CNN_LSTM', 'finetuning_CNN_LSTM_naive']
            if 'naive' in model_type:
                model_trained_data_split = 'orig_all_edit_10'
                model_type = model_type.strip('_naive')
                #ipdb.set_trace()
            else:
                model_trained_data_split = 'orig_all' #'orig_all_edit_10'   # 'orig_all'


        else:
            if model_type == 'finetuning_CNN_LSTM' or model_type == 'finetuning_SAAA' or model_type == 'finetuning_SAAA_lr_e-5' or model_type == 'finetuning_SAAA_lr_e-7':
                model_trained_data_split = 'orig_10'

            elif model_type == 'finetuning_CNN_LSTM_naive':
                model_trained_data_split = 'orig_10_edit_10'
                model_type = 'finetuning_CNN_LSTM'
            elif model_type == 'finetuning_SAAA_naive':
                model_trained_data_split = 'orig_10_edit_10'
                model_type = 'finetuning_SAAA'
            elif model_type == 'finetuning_SAAA_lr_e-5_naive':
                model_trained_data_split = 'orig_10_edit_10'
                model_type = 'finetuning_SAAA_lr_e-5'



            elif model_type == 'finetuning_CNN_LSTM_naive_cnt_orig_del' or model_type =='finetuning_CNN_LSTM_naive_cnt_orig_edit_del':   ## test type orig_10 and del_! comparison
                if model_type == 'finetuning_CNN_LSTM_naive_cnt_orig_del':
                    model_trained_data_split = 'orig_10_del1'         ## if counting: orig_10_edit_10_del1; orig_10_del1
                if  model_type == 'finetuning_CNN_LSTM_naive_cnt_orig_edit_del':
                    model_trained_data_split  = 'orig_10_edit_10_del1'
                model_type = 'finetuning_CNN_LSTM'


            elif model_type == 'finetuning_SAAA_naive_cnt_orig_del' or model_type =='finetuning_SAAA_naive_cnt_orig_edit_del' or \
                model_type == 'finetuning_SAAA_lr_e-5_naive_cnt_orig_del' or model_type == 'finetuning_SAAA_lr_e-5_naive_cnt_orig_edit_del' or \
                model_type == 'finetuning_SAAA_lr_e-7_naive_cnt_orig_del' or model_type == 'finetuning_SAAA_lr_e-7_naive_cnt_orig_edit_del':
                if 'naive_cnt_orig_del' in model_type:
                    model_trained_data_split = 'orig_10_del1'         ## if counting: orig_10_edit_10_del1; orig_10_del1
                if  'naive_cnt_orig_edit_del' in model_type:
                    model_trained_data_split  = 'orig_10_edit_10_del1'
                if 'lr_e-5' in model_type:
                    model_type = 'finetuning_SAAA_lr_e-5'
                elif 'lr_e-7' in model_type:
                    model_type = 'finetuning_SAAA_lr_e-7'
                else:
                    model_type = 'finetuning_SAAA'
            else:
                model_trained_data_split = 'orig_10_edit_10'   # 'orig_all_edit_10'



        if 'CNN_LSTM' in model_type:
            net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
        elif 'SAAA' in model_type:  # args.model_type == 'finetuning_SAAA' or args.model_type == 'data_aug_SAAA':
            net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()




        model_path_folder = os.path.join('./models/' + model_type + '/' + config.ques_type + '/' + model_trained_data_split)
        model_path = os.path.join(model_path_folder, 'epoch_{}.pth'.format(epoch_list[idx]))


        if args.model_type_no_ft_epoch00:  ### geting epoch 00 model....results
            if 'CNN_LSTM' in model_type:
                model_path = './models/no_attn_biasTrue.pth'    # CNN+LSTM
                res_json_folder = os.path.join('/BS/vedika3/nobackup/pytorch-vqa/finetuning_logs/EPOCH_00_cnn_lstm/' + config.ques_type + '/entire_logs_using_epoch-1/', config.test_data_split)
            if 'SAAA' in model_type:
                model_path = './models/show_ask_attend_answer.pth' # SAAA
                res_json_folder = os.path.join('/BS/vedika3/nobackup/pytorch-vqa/finetuning_logs/EPOCH_00_SAAA/' +config.ques_type+ '/entire_logs_using_epoch-1/', config.test_data_split)

            os.makedirs(res_json_folder, exist_ok=True)
            res_pkl = os.path.join(res_json_folder, 'results_fineuned_using_no_finetuning_epoch_00_model.pickle')

        else:
            res_json_folder = os.path.join('/BS/vedika3/nobackup/pytorch-vqa/finetuning_logs/' +  model_type + '/' + config.ques_type + '/entire_logs_using_epoch' + str(epoch_list[idx]) +  '/' + config.test_data_split)
            os.makedirs(res_json_folder, exist_ok=True)
            res_json = os.path.join(res_json_folder, 'results_fineuned_using_' + model_trained_data_split + '.json')
            res_pkl = os.path.join(res_json_folder, 'results_fineuned_using_' + model_trained_data_split + '.pickle')

        print('loading model from', model_path)
        net.load_state_dict(torch.load(model_path)["weights"])   ### so here you load the weights, essentially the model
        #print(net)
        net.eval()

        output_qids_answers = []
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
            print('#IQAs tested:', len(output_qids_answers))
            #ipdb.set_trace()
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
