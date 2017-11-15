#!/bin/bash

#SBATCH --mem=16000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=copy.out    # %N for node name, %j for jobID

rsync -P -r /home/edward/project/edward/ctcolonography /home/rbbidart/project/rbbidart/