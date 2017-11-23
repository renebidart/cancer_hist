#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-23:00            # time (DD-HH:MM)
#SBATCH --output=%N-%j.out    # %N for node name, %j for jobID model_conv_incp3_128.out

module load cuda cudnn python/3.5.2
source tensorflow/bin/activate

python /home/rbbidart/cancer_hist/src/reg_test.py project/rbbidart/cancer_hist/full_slides /home/rbbidart/cancer_hist/output/reg_conv2_100 200 8 100 conv2
