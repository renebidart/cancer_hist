#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=64000M               # memory per node
#SBATCH --time=0-18:00            # time (DD-HH:MM)
#SBATCH --output=crc_incp3_128.out    # %N for node name, %j for jobID model_conv_incp3_128.out

module load cuda cudnn python/3.5.2
source tensorflow3/bin/activate

python /home/rbbidart/cancer_hist/src/crc_train.py /home/rbbidart/project/rbbidart/cancer_hist/crc_128 /home/rbbidart/cancer_hist_out/crc/incp_128 120 16 128 conv_incp3
