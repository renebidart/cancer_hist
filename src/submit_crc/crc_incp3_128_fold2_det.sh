#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=crc_incp3_128_f2_det.out    # %N for node name, %j for jobID

module load cuda cudnn python/3.5.2
source tensorflow3/bin/activate

python /home/rbbidart/cancer_hist/src/crc_train.py /home/rbbidart/project/rbbidart/cancer_hist/crc_2_fold_128_det/crc_128_fold_2_det /home/rbbidart/cancer_hist_out/crc/incp3_128_f2_det 120 16 128 conv_incp3
