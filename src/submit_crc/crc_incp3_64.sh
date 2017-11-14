#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-18:00            # time (DD-HH:MM)
#SBATCH --output=crc_incp3_64.out    # %N for node name, %j for jobID

module load cuda cudnn python/3.5.2
source tensorflow4/bin/activate

python /home/rbbidart/cancer_hist/src/crc_train.py /home/rbbidart/project/rbbidart/cancer_hist/crc_64 /home/rbbidart/cancer_hist_out/crc/incp3_64 120 16 64 conv_incp3
