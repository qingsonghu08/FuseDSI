# ViT-S+ICS
#CUDA_VISIBLE_DEVICES=2,3 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 --conv-stem -pp ../../model/vit_small_ics_cfs_lup.pth --logs-dir ../../log/cluster_contrast_reid/msmt17/vit_small_ics_cfs_lup --eval-step 50

# VIT-S
# CUDA_VISIBLE_DEVICES=0 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp ../../model/vit_small_cfs_lup.pth --logs-dir ../../log/cluster_contrast_reid/msmt17/vit_small_cfs_lup --eval-step 50
# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp /mnt1/michuan.lh/log/dino/lup_filter/deit_small_251w_forget/checkpoint.pth --logs-dir ../../log/cluster_contrast_reid/msmt17/vit_small_cfs_lup/checkpoint --eval-step 50

CUDA_VISIBLE_DEVICES=1 python train.py -d msmt --iters 200 --eps 0.7 --num-instances 8  --eval-step 10 --Lambda1 1.0 --Lambda2 5.0 --nncl --rcl --gcn --knn 5 --gcnk1 5 --gcnk2 3 \
--logs-dir log_sota_msmt1  --resume /home/a/data/hqs_2023_v3/code5/cluster-contrast-reid-v2/log_base_msmt/checkpoint30.pth.tar

