import h5py
import ipdb

filename_orig = '/BS/vedika3/nobackup/pytorch-vqa/preprocessed_data/trainset.h5'
f_orig = h5py.File(filename_orig, 'r')
data_ids_orig = f_orig['ids'][()]
data_feats_orig = list(f_orig['features'])


filename = '/BS/vedika3/nobackup/pytorch-vqa/preprocessed_data/edit_trainset.h5'
f = h5py.File(filename, 'r')
data_ids = f['ids'][()]
data_feats = list(f['features'])


ipdb.set_trace()