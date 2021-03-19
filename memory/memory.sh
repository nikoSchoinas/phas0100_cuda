#!/bin/bash -l

# Bash script to run the CUDA example on Myriad

# Request one GPU card (out of the maximum of two)
#$ -l gpu=1

# Request a wall clock time limit of 1 minute (format hours:minutes:seconds)
#$ -l h_rt=0:1:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G or T)
#$ -l mem=1G

# Request 1 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=1G

# Set the name of the job.
#$ -N memory

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/ucapnsc/phas0100/cuda/phas0100_cuda/memory/output

# load the cuda module (as we are running a CUDA program)
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2

# Run the application
../build/memory
