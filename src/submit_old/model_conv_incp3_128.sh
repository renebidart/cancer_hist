#!/bin/bash

#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-24:00            # time (DD-HH:MM)
#SBATCH --output=%N-%j.out    # %N for node name, %j for jobID model_conv_incp3_128.out

module load cuda cudnn python/3.5.2
source tensorflow/bin/activate

python /home/rbbidart/cancer_hist/src/model_test.py project/rbbidart/cancer_hist/extracted_cells_128 /home/rbbidart/cancer_hist/output/size128_class 80 128 128 conv_incp3