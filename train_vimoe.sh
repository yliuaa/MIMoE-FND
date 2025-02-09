# ---------------------------------------------------------------------------- #
#                                    Weibo21                                   #
# ---------------------------------------------------------------------------- #
# CUDA_VISIBLE_DEVICES=1 python ./train_vimoe.py -train_dataset weibo21 \
#                                         -test_dataset weibo21 \
#                                         -batch_size 24 \
#                                         -epochs 30 \
#                                         -val 0 \
#                                         -duplicate_fake_times 0 \
#                                         -device cuda:0 \
#                                         -int_lr 1e-4 \
#                                         -int_beta 0.7 \
#                                         -agr_threshold 0.3 \
#                                         -sem_threshold 0.3 \
#                                         -note "train"

# inference
CUDA_VISIBLE_DEVICES=1 python ./train_vimoe.py -train_dataset weibo21 \
                                        -test_dataset weibo21 \
                                        -val 1 \
                                        -batch_size 16 \
                                        -epochs 50 \
                                        -checkpoint /home/yifan40/archive/MIMoE-FND/checkpoints/weibo21/31_1119_95.pkl \
                                        -get_MLP_score 0 \
                                        -device cuda:0 \
                                        -note "val"




# ---------------------------------------------------------------------------- #
#                                    Weibo                                     #
# ---------------------------------------------------------------------------- #
# CUDA_VISIBLE_DEVICES=1 python ./train_vimoe.py -train_dataset weibo \
#                                         -test_dataset weibo \
#                                         -batch_size 24 \
#                                         -epochs 50 \
#                                         -checkpoint /home/yifan40/archive/MIMoE-FND/checkpoints/weibo/30_1119_92.pkl \
#                                         -device cuda:0 \
#                                         -val 1 \
#                                         -get_MLP_score 0 \



# CUDA_VISIBLE_DEVICES=1 python ./train_vimoe.py -train_dataset weibo \
#                                         -test_dataset weibo \
#                                         -batch_size 24 \
#                                         -epochs 50 \
#                                         -val 0 \
#                                         -duplicate_fake_times 0 \
#                                         -not_on_12 1 \
#                                         -device cuda:0 \
#                                         -int_lr 1e-6\
#                                         -int_beta 0.3 \
#                                         -note "int_LR Check"

# ---------------------------------------------------------------------------- #
#                                   GossipCop                                  #
# ---------------------------------------------------------------------------- #
# #Inference
# CUDA_VISIBLE_DEVICES=0 python ./train_vimoe.py -train_dataset gossip \
#                                         -test_dataset gossip \
#                                         -batch_size 10 \
#                                         -epochs 50 \
#                                         -checkpoint /home/yifan40/archive/MIMoE-FND/checkpoints/gossip/33_1118_89.pkl \
#                                         -device cuda:0 \
#                                         -val 1 \
#                                         -get_MLP_score 0


# Train
# CUDA_VISIBLE_DEVICES=0 python ./train_vimoe.py -train_dataset gossip \
#                                         -test_dataset gossip \
#                                         -batch_size 24 \
#                                         -epochs 50 \
#                                         -val 0 \
#                                         -duplicate_fake_times 0 \
#                                         -not_on_12 1 \
#                                         -device cuda:0 \
#                                         -int_lr 1e-6 \
#                                         -int_beta 0.1 \
#                                         -agr_threshold 0.3 \
#                                         -sem_threshold 0.3 \
#                                         -note "train" 

