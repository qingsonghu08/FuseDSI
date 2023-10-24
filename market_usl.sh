# Market1501
CUDA_VISIBLE_DEVICES=0 python train.py -d market --iters 200 --eps 0.6 --num-instances 8 --eval-step 5 --Lambda1 1.0 --Lambda2 5.0 --nncl --rcl --gcn --knn 5 --gcnk1 4 --gcnk2 2 \
--logs-dir log_sota_market1 --resume /home/a/data/hqs_2023_v3/code3/TransReID-SSL-main-v1/cluster-contrast-reid/log_base_market/checkpoint30.pth.tar

# MSMT17
#CUDA_VISIBLE_DEVICES=1 python train_msmt.py -d msmt --iters 200 --eps 0.7 --num-instances 8  --eval-step 10 --Lambda1 1.0 --Lambda2 5.0 --nncl --rcl --gcn --knn 5 --gcnk1 5 --gcnk2 3 \
#--logs-dir log_sota_msmt1  --resume /home/a/data/hqs_2023_v3/code5/cluster-contrast-reid-v2/log_base_msmt/checkpoint30.pth.tar


