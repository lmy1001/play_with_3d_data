#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=40G
#$ -q gpu.24h.q

/itet-stor/menliu/net_scratch/conda_envs/pytcu10/bin/python -u train_pointnet_cls.py "$@"
