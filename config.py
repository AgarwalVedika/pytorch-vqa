# preprocess config
preprocess_batch_size = 32
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
preprocessed_path = './preprocessed_data/orig_edit_VQAset.h5'
train_path = './mscoco/train2014'  # directory of training images
val_path = './mscoco/val2014'  # directory of validation images
test_path = './mscoco/test2015'  # directory of test images
edit_train_path =  './edit_mscoco/train2014'
edit_val_path =  './edit_mscoco/val2014'

# training config
epochs = 50
batch_size = 128   ## originally 128 for learning
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000

#train.path
orig_edit_equal_batch = 0    ## default it is zero- only when you use different data aug styles (2/3)
orig_edit_diff_ratio_naive = 0
orig_edit_diff_ratio_naive_no_edit_ids_repeat = 0
regulate_old_loss = 0
load_only_orig_ids = 0
enforce_consistency = 0
model_path_no_attn = '/BS/vedika3/nobackup/pytorch-vqa//models/no_attn_biasTrue.pth'     # biasTrue better than biasFalse by ~0.3 points!
model_path_show_ask_attend_answer = './models/show_ask_attend_answer.pth'   ### the one initially with everything
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
dset = 'v2'   ## change to v1 if you want to evaluate on VQA v1- TODO changes to be made in the preprocess-vocab then- to hndle......
task = 'OpenEnded'
dataset = 'mscoco'

### FINETUNIGNNEXPERIMENT

## training phase


# make sure train.py and data.py: edit_orig_combine is true
ques_type = 'what color is the'  # ['how many', 'is this a', 'is there a', 'what color is the', 'counting']
data_split = 'orig_10_edit_10'  # ['orig_10', 'orig_all', 'orig_10_edit_10','orig_all_edit_10',  'orig_all_edit_all']
# #
# # ###SETTING_2
edit_loader_type = 'get_edits'  # 'get_more_edits_if_not_64' , get_edits_if_not_orig , get_edits , get_all_edits
orig_amt = 0.66
load_only_orig_ids = 1
enforce_consistency = 1
regulate_old_loss = 1
lam_edit_loss = 0.5
lam_CE = 0.5
lam_KL = 0.5
lam_MSE = 0.5
model_type =  'finetuning_CNN_LSTM_data_aug2_{}_origamt_{}'.format(edit_loader_type, orig_amt)           #'finetuning_CNN_LSTM'  #finetuning_CNN_LSTM_data_aug2
orig_edit_equal_batch = 1
orig_edit_diff_ratio_naive = 0
orig_edit_diff_ratio_naive_no_edit_ids_repeat = 0
trained_model_save_folder = './models/' + model_type  + '/' + ques_type  + '/' + data_split # os.path.join('./models', ques_type)
qa_path = '/BS/vedika2/nobackup/thesis/mini_datasets_qa_CNN_finetune_training' + '/'+ ques_type + '/' + data_split  # directory containing the question and annotation jsons


# SETTING 3
# #model_type =  'finetuning_CNN_LSTM_ratio_mix_no_repeat_edit_ids_orig:edit_{}:{}'.format(orig_amt, edit_amt)
# ## playing with different ratio type - 2 loaders: train separate, edit_separate
# orig_amt = 0.7
# edit_amt = 0.3
#load_only_orig_ids = 0
# model_type =  'finetuning_CNN_LSTM_ratio_mix_no_repeat_edit_ids_orig:edit_{}:{}'.format(orig_amt, edit_amt)           #'finetuning_CNN_LSTM'  #finetuning_CNN_LSTM_data_aug2
# orig_edit_equal_batch = 0
# orig_edit_diff_ratio_naive = 0
# orig_edit_diff_ratio_naive_no_edit_ids_repeat = 1   # ( only one of these 3 could be one)
# trained_model_save_folder = './models/' + model_type  + '/' + ques_type  + '/' + data_split # os.path.join('./models', ques_type)
# qa_path = '/BS/vedika2/nobackup/thesis/mini_datasets_qa_CNN_finetune_training' + '/'+ ques_type + '/' + data_split  # directory containing the question and annotation jsons



# ##SETTING 4
# ## playing with different ratio type - 2 loaders: train separate, edit_separate
# orig_amt = 0.5
# edit_amt = 0.5
# model_type =  'finetuning_CNN_LSTM_ratio_mix_load_only_orig_ids_no_repeat_edit_ids_orig:edit_{}:{}'.format(orig_amt, edit_amt)           #'finetuning_CNN_LSTM'  #finetuning_CNN_LSTM_data_aug2
# orig_edit_equal_batch = 0
# load_only_orig_ids = 1
# orig_edit_diff_ratio_naive = 0
# orig_edit_diff_ratio_naive_no_edit_ids_repeat = 1   # ( only one of these 3 could be one)
# trained_model_save_folder = './models/' + model_type  + '/' + ques_type  + '/' + data_split # os.path.join('./models', ques_type)
# qa_path = '/BS/vedika2/nobackup/thesis/mini_datasets_qa_CNN_finetune_training' + '/'+ ques_type + '/' + data_split  # directory containing the question and annotation jsons




# #
# # # # testing phase
# model_type =  'finetuning_CNN_LSTM_ratio_mix_orig:edit_0.3:0.7'
# ques_type = 'what color is the'  # ['how many', 'is this a', 'is there a', 'what color is the', 'counting']
# test_data_split = 'edit_all' #['orig_90_10',  'orig_90_all', 'edit_10', 'edit_all']    #'orig_10_10' was for validation for all 50 epoch models
# ft_logs_folder = '/BS/vedika3/nobackup/pytorch-vqa/finetuning_logs_diff_str_using_epoch49/' + model_type  + '/' + ques_type  + '/' + test_data_split # os.path.join('./models', ques_type)
# # paths
# qa_path = '/BS/vedika2/nobackup/thesis/mini_datasets_qa_CNN_finetune_testing' + '/'+ ques_type + '/' + test_data_split  # directory containing the question and annotation jsons
# ###ques-type_vall_10_all chosen epochs:
# all_49_dict = {'how_many_orig_10': 49,
#  'how_many_orig_all': 49,
#  'how_many_orig_10_edit_10': 49,
#  'how_many_orig_all_edit_10': 49,
#  'how_many_orig_all_edit_all': 49,
#  'is_this_a_orig_10': 49,
#  'is_this_a_orig_all': 49,
#  'is_this_a_orig_10_edit_10': 49,
#  'is_this_a_orig_all_edit_10': 49,
#  'is_this_a_orig_all_edit_all': 49,
#  'is_there_a_orig_10': 49,
#  'is_there_a_orig_all': 49,
#  'is_there_a_orig_10_edit_10': 49,
#  'is_there_a_orig_all_edit_10': 49,
#  'is_there_a_orig_all_edit_all': 49,
#  'what_color_is_the_orig_10': 49,
#  'what_color_is_the_orig_all': 49,
#  'what_color_is_the_orig_10_edit_10': 49,
#  'what_color_is_the_orig_all_edit_10': 49,
#  'what_color_is_the_orig_all_edit_all': 49,
#  'counting_orig_10': 49,
#  'counting_orig_all': 49,
#  'counting_orig_10_edit_10': 49,
#  'counting_orig_all_edit_10': 49,
#  'counting_orig_all_edit_all': 49}




# ft_val_10_naive_CNN_LSTM = {'how_many_orig_10': 46,   #ft_val_10
#  'how_many_orig_all': 1,
#  'how_many_orig_10_edit_10': 18,
#  'how_many_orig_all_edit_10': 2,
#  'how_many_orig_all_edit_all': 5,
#  'is_this_a_orig_10': 14,
#  'is_this_a_orig_all': 21,
#  'is_this_a_orig_10_edit_10': 0,
#  'is_this_a_orig_all_edit_10': 0,
#  'is_this_a_orig_all_edit_all': 5,
#  'is_there_a_orig_10': 22,
#  'is_there_a_orig_all': 13,
#  'is_there_a_orig_10_edit_10': 33,
#  'is_there_a_orig_all_edit_10': 42,
#  'is_there_a_orig_all_edit_all': 17,
#  'what_color_is_the_orig_10': 5,
#  'what_color_is_the_orig_all': 2,
#  'what_color_is_the_orig_10_edit_10': 0,
#  'what_color_is_the_orig_all_edit_10': 21,
#  'what_color_is_the_orig_all_edit_all': 2,
#  'counting_orig_10': 27,
#  'counting_orig_all': 11,
#  'counting_orig_10_edit_10': 20,
#  'counting_orig_all_edit_10': 1,
#  'counting_orig_all_edit_all': 3}


ft_val_10_naive_CNN_LSTM_edit_more15 = {'how_many_orig_10': 46,
 'how_many_orig_all': 1,
 'how_many_orig_10_edit_10': 18,
 'how_many_orig_all_edit_10': 21,
 'how_many_orig_all_edit_all': 16,
 'is_this_a_orig_10': 14,
 'is_this_a_orig_all': 21,
 'is_this_a_orig_10_edit_10': 45,
 'is_this_a_orig_all_edit_10': 19,
 'is_this_a_orig_all_edit_all': 28,
 'is_there_a_orig_10': 22,
 'is_there_a_orig_all': 13,
 'is_there_a_orig_10_edit_10': 33,
 'is_there_a_orig_all_edit_10': 42,
 'is_there_a_orig_all_edit_all': 17,
 'what_color_is_the_orig_10': 5,
 'what_color_is_the_orig_all': 2,
 'what_color_is_the_orig_10_edit_10': 44,
 'what_color_is_the_orig_all_edit_10': 21,
 'what_color_is_the_orig_all_edit_all': 16,
 'counting_orig_10': 27,
 'counting_orig_all': 11,
 'counting_orig_10_edit_10': 20,
 'counting_orig_all_edit_10': 16,
 'counting_orig_all_edit_all': 22}



# ft_val_10_naive_SAAA = {'how_many_orig_10': 2,
#  'how_many_orig_all': 0,
#  'how_many_orig_10_edit_10': 0,
#  'how_many_orig_all_edit_10': 2,
#  'how_many_orig_all_edit_all': 8,
#  'is_this_a_orig_10': 2,
#  'is_this_a_orig_all': 32,
#  'is_this_a_orig_10_edit_10': 0,
#  'is_this_a_orig_all_edit_10': 10,
#  'is_this_a_orig_all_edit_all': 19,
#  'is_there_a_orig_10': 15,
#  'is_there_a_orig_all': 14,
#  'is_there_a_orig_10_edit_10': 11,
#  'is_there_a_orig_all_edit_10': 24,
#  'is_there_a_orig_all_edit_all': 3,
#  'what_color_is_the_orig_10': 2,
#  'what_color_is_the_orig_all': 0,
#  'what_color_is_the_orig_10_edit_10': 0,
#  'what_color_is_the_orig_all_edit_10': 25,
#  'what_color_is_the_orig_all_edit_all': 18,
#  'counting_orig_10': 0,
#  'counting_orig_all': 28,
#  'counting_orig_10_edit_10': 0,
#  'counting_orig_all_edit_10': 3,
#  'counting_orig_all_edit_all': 2}



# #
# # ** config_orig_vqa **
# # paths
# qa_path = 'vqa'  # directory containing the question and annotation jsons
# ##eval.py
# results_with_attn_pth = './logs/train_with_attn_.pth'                          ### change this!! train/val/test!
# results_with_attn_pkl = './logs/train_with_attn_.pickle'
# results_no_attn_pth = './logs/train_no_attn_.pth'
# results_no_attn_pkl = './logs/train_no_attn_.pickle'



# # ######################edited_vqa_
# # #** config_edit_vqa **
# # ##paths
# qa_path = 'edit_vqa'  # directory containing the question and annotation jsons
#
# ##eval.py
# results_with_attn_pth =  './logs/edit_train_with_attn_.pth'        ### change this!! train/val/test!
# results_with_attn_pkl = './logs/edit_train_with_attn_.pickle'
# results_no_attn_pth = './logs/edit_train_no_attn_.pth'
# results_no_attn_pkl = './logs/edit_train_no_attn_.pickle'
# ########################edited_vqa_
# # ** config_edit_vqa **




######################edited_vqa_flipped
#** config_edit_vqa **
##paths
#qa_path = 'edit_vqa'  # directory containing the question and annotation jsons

#train_path = './edit_mscoco_flipped/train2014'  # directory of training images
#val_path = './edit_mscoco_flipped/val2014'  # directory of validation images
#test_path = './edit_mscoco/test2015'

#preprocessed_path = './preprocessed_data/flipped_edit_valset.h5'      ### THIS IS USED IN THE CODE

##eval.py
#results_with_attn_pth = './logs/flipped_edit_val_with_attn_.pth'
#results_with_attn_pkl = './logs/flipped_edit_val_with_attn_.pickle'
#results_no_attn_pth = './logs/flipped_edit_val_no_attn_.pth'
#results_no_attn_pkl = './logs/flipped_edit_val_no_attn_.pickle'
#########################edited_vqa_
#** config_edit_vqa **




# # ##SPORTS_WITHOUT_PERSON##
# # ** config_orignal val for eidted mini-json files: sports_without_person **
# # paths
# qa_path = '/BS/vedika2/nobackup/thesis/mini_datasets_qa/sports_without_person'  # directory containing the question and annotation jsons

# ##eval.py
# results_with_attn_pth = './logs/sports_wihtout_person_with_attn_.pth'
# results_with_attn_pkl = './logs/sports_wihtout_person_with_attn_.pickle'
# results_no_attn_pth = './logs/sports_wihtout_person_no_attn_.pth'
# results_no_attn_pkl = './logs/sports_wihtout_person_no_attn_.pickle'
#
#


