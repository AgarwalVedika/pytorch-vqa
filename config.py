# ** config_orig_vqa **
# paths
qa_path = 'vqa'  # directory containing the question and annotation jsons

train_path = './mscoco/train2014'  # directory of training images
val_path = './mscoco/val2014'  # directory of validation images
#test_path = './mscoco/test2015'  # directory of test images

preprocessed_path = './preprocessed_data/resnet-14x14.h5'   ### train + val + test
preprocessed_train_path = './preprocessed_data/trainset.h5'  # path where preprocessed features are saved to and loaded from
preprocessed_val_path = './preprocessed_data/valset.h5'  # path where preprocessed features are saved to and loaded from
preprocessed_test_path = './preprocessed_data/testset.h5'  # path where preprocessed features are saved to and loaded from

##eval.py
results_with_attn_pth = './logs/val_with_attn_.pth'
results_with_attn_pkl = './logs/val_with_attn_.pickle'
results_no_attn_pth = './logs/val_no_attn_.pth'
results_no_attn_pkl = './logs/val_no_attn_.pickle'

#######################edited_vqa_
# #** config_edit_vqa **
#paths
# qa_path = 'edit_vqa'  # directory containing the question and annotation jsons
#
# train_path = './edit_mscoco/train2014'  # directory of training images
# val_path = './edit_mscoco/val2014'  # directory of validation images
# #test_path = './edit_mscoco/test2015'
#
# preprocessed_path = './preprocessed_data/edit_valset.h5'           ##edit_resnet-14x14.h5'   ### train + val + test
# preprocessed_train_path = './preprocessed_data/edit_trainset.h5'  # path where preprocessed features are saved to and loaded from
# preprocessed_val_path = './preprocessed_data/edit_valset.h5'  # path where preprocessed features are saved to and loaded from
# preprocessed_test_path = './preprocessed_data/edit_testset.h5'  # path where preprocessed features are saved to and loaded from
#
# ##eval.py
# results_with_attn_pth = './logs/edit_val_with_attn_.pth'
# results_with_attn_pkl = './logs/edit_val_with_attn_.pickle'
# results_no_attn_pth = './logs/edit_val_no_attn_.pth'
# results_no_attn_pkl = './logs/edit_val_no_attn_.pickle'
# #########################edited_vqa_
# ## ** config_edit_vqa **

#train.path
model_path_no_attn = './models/no_attn_biasFalse_norelu.pth'                     ### originally bias False, drop=0.5
# train E049: 100% 3396/3396 [06:55<00:00,  8.18it/s, acc=0.7492, loss=1.2638]
# val E049: 100% 1675/1675 [03:24<00:00,  8.20it/s, acc=0.5203, loss=1.7672]
model_path_show_ask_attend_answer = './models/show_ask_attend_answer.pth'   ### the one initially with everything

vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to

dset = 'v2'   ## change to v1 if you want to evaluate on VQA v1- TODO changes to be made in the preprocess-vocab then- to hndle......
task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 50
batch_size = 128
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000
