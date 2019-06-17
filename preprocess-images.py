import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

import config
import data
import utils
from resnet import resnet as caffe_resnet
import argparse
import ipdb
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True, type=str)  ### train2014/val2014/test2015/edit_val201
parser.add_argument('--edit_set', required=True, type=bool)  ## True False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_coco_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    #ipdb.set_trace()  ## datasets[0].__getitem__(116591)[0]  print the largest coco_id - this is within torch int64 bound!
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main(args):  # main(args):

    cudnn.benchmark = True

    net = Net().cuda()
    net.eval()

    if args.split == 'train2014':                                   ## edited to handle different splits - user can give the command now
        loader_path = config.train_path
        dump_path = config.preprocessed_train_path
    if args.split == 'val2014':
        loader_path = config.val_path
        dump_path = config.preprocessed_val_path
    if args.split == 'test2015':
        loader_path = config.test_path
        dump_path = config.preprocessed_test_path
    # if args.split == 'edit_val2014':
    #     loader_path = config.edit_val_path
    #     dump_path = config.preprocessed_edit_val_path

    #loader = create_coco_loader(config.train_path, config.val_path, config.test_path)
    loader = create_coco_loader(loader_path)    #(loader_path)
    #ipdb.set_trace()
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(dump_path, libver='latest') as fd:    ##(with h5py.File(config.preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        if args.edit_set == True:
            coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype="S25")    ## TODO handling edit set fix for string image ids
        else:
            coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(loader):
            #imgs = Variable(imgs.cuda(async=True), volatile=True)
            imgs = Variable(imgs.cuda(async=True))
            with torch.no_grad():
                out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            #ipdb.set_trace()
            if args.edit_set == True:
                coco_ids[i:j] = numpy.string_(ids)        ####TODO vedika  for edit set this is it  # for numpy.string_ dtype='S25'
            else:
                coco_ids[i:j] = ids.numpy().astype('int')  # ### TODO vedika made it iimage_id) : for future purpose- maybe a good idea to id edited images as int and not str
            i = j


if __name__ == '__main__':
    args = parser.parse_args()
    if args.edit_set:
        print('reminder to check if right config')
    main(args)
