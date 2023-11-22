hostname
python -u main.py --lr 0.05 --epochs 1000 --dataset isic \
                --arch resnet50 --use_norm True \
                --wd 2e-4 --cos True --cl_views sim-sim \
                --workers 32 --batch-size 110 \
                --alpha 1.0 --beta 0.35 --ce_loss 'WeightedBCE' --loss_req 'BCL' \
                --user_name '' --pretrained True \
                --data "" --val_data "" --txt "" --val_txt ""