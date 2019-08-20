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
import model2   ## modeifed net to have no attention
import utils
import time
import argparse
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default= 'finetuning_CNN_LSTM_data_aug2', type=str)  ## 'with_attn' , 'no_attn', 'finetuning_SAAA'  'finetuning_CNN_LSTM_data_aug2'



def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0

def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0, dataset=None):
    """ Run an epoch over the given loader """
    if train:
        net.train()

        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        # answ = []   # edit_vedika FT101
        # idxs = []
        # accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax(dim=1).cuda()   ### nn.LogSoftmax().cuda()

    for batch in tq:                                            #for v, q, a, idx, img_id, ques_id, q_len in tq:
        v, q, a, idx, img_id, ques_id, q_len = batch
        #  except image_id- everything is a tensor   ## [i[0].dtype for i in [v, ques, ans, idx,  ques_id, q_len]]
        ## [v, ques, ans, idx,  ques_id, q_len].dtype = [torch.float32, torch.int64, torch.float32, torch.int64, torch.int64, torch.int64]

        if config.orig_edit_equal_batch:
            #edit_v, edit_q, edit_a, edit_idx, edit_img_id, edit_ques_id, edit_q_len = data.get_edit_train_loader(ques_id_batch=ques_id, train=True)
            edit_batch = data.get_edit_train_batch(dataset=dataset, ques_id_batch=ques_id, train=True)
            #[torch.float32, torch.int64, torch.float32, torch.int64, [torch.int64, torch.int64]]
            v, q, a, idx,  ques_id, q_len =  [torch.cat((batch[i], edit_batch[i]), dim=0) for i in [0,1,2,3,5,6]]
            img_id = img_id + edit_batch[4]  # edit_img_id = edit_batch[4]

        ipdb.set_trace()

        var_params = {
            #'volatile': not train,  # taken care for val using: with torch.no_grad():
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        if train:
            out = net(v, q, q_len)
            nll = -log_softmax(out)  ## taking softmax here
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out.data, a.data).cpu()
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iterations += 1

            ipdb.set_trace()

        else:
            with torch.no_grad():
                out = net(v, q, q_len)
                nll = -log_softmax(out)  ## taking softmax here
                loss = (nll * a / 10).sum(dim=1).mean()
                acc = utils.batch_accuracy(out.data, a.data).cpu()   ### taking care of volatile=True for val

            # # store information about evaluation of this minibatch     # edit_vedika FT101
            # _, answer = out.data.cpu().max(dim=1)
            # answ.append(answer.view(-1))
            # accs.append(acc.view(-1))
            # idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())    #data[0])
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    # if not train:               ## # edit_vedika FT101
    #     answ = list(torch.cat(answ, dim=0))
    #     accs = list(torch.cat(accs, dim=0))
    #     idxs = list(torch.cat(idxs, dim=0))
    #     return answ, accs, idxs



def main(args):
    start_time = time.time()

    cudnn.benchmark = True

    train_dataset, train_loader = data.get_loader(train=True)
    _, val_loader = data.get_loader(val=True)
    #test_loader = data.get_loader(test=True)

    if args.model_type == 'no_attn':
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(config.model_path_no_attn)
    elif args.model_type == 'with_attn':
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(config.model_path_show_ask_attend_answer)

    elif args.model_type == 'finetuning_CNN_LSTM' or args.model_type =='finetuning_CNN_LSTM_data_aug2':
        net = nn.DataParallel(model2.Net(val_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_no_attn)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(config.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif args.model_type == 'finetuning_SAAA' or args.model_type =='finetuning_SAAA_data_aug2':
        net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_show_ask_attend_answer)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(config.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

        # os.makedirs(target_name, exist_ok=True)
    print('will save to {}'.format(target_name))

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i, dataset=train_dataset)   ## prefix needed as ths is passed to tracker- which stroes then train_acc/loss
        _ = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)    ## prefix needed as ths is passed to tracker- which stroes then val acc/loss

        results = {
            'tracker': tracker.to_dict(),   ## tracker saves acc/loss for all 50 epochs- since it appends the values ( lines 91..)
            'config': config_as_dict,
            'weights': net.state_dict(),
            # 'eval': {          ## # edit_vedika FT101 you are svaing the results here - you dont need this!
            #     'answers': r[0],
            #     'accuracies': r[1],
            #     'idx': r[2],
            # },
            'vocab': train_loader.dataset.vocab,
        }
        saving_target_name = 'epoch_{}.pth'.format(i)   ## you want to have all finetuned models- so save every model at everye epoch
        torch.save(results, os.path.join(target_name, saving_target_name))   ## keys:  "name", "tracker", "config", "weights", "eval", "vocab"

    #checkpoint_file = os.path.join('./models/epoch_{}.pth'.format(i))
    #torch.save(net, checkpoint_file)
    #print('saving model to '+ checkpoint_file)

    print('time_taken:', time.time() - start_time)

if __name__ == '__main__':
    args = parser.parse_args()
    assert config.model_type == args.model_type
    main(args)
