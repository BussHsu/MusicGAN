#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 16 # embedding dimension
HIDDEN_DIM = 64 # hidden state dimension of lstm cell

START_TOKEN = 0
PRE_EPOCH_NUM = 200 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = EMB_DIM
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
MAX_NUM_IN_EPOCH = 12000    # test on 8000 instance to get test loss
TOTAL_BATCH = 80
SAMP_NUM = 50
#src file
positive_file = 'Data/all_reel.dat'
eval_real_file = 'Data/all_reel_eval.dat'
#tgt file
negative_file = 'Data/generator_sample.dat'
eval_file = 'Data/eval.dat'
generated_num = MAX_NUM_IN_EPOCH
##########################################################################################
# Data Params
##########################################################################################
SEQ_LENGTH = 30 # sequence length
vocab_size = 95