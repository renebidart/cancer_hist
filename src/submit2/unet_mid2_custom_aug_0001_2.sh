#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-18:00            # time (DD-HH:MM)
#SBATCH --output=unet_mid2_custom_aug_0001_2-%N-%j.out    # %N for node name, %j for jobID model_conv_incp3_128.out

module load cuda cudnn python/3.5.2
source tensorflow6/bin/activate

python /home/rbbidart/cancer_hist/src/unet_dist_test.py /home/rbbidart/project/rbbidart/cancer_hist/full_slides2 /home/rbbidart/project/rbbidart/cancer_hist/im_dist_labels /home/rbbidart/cancer_hist_out/unet_dist/unet_mid2_custom_aug_0001_2 100 4 unet_mid2

