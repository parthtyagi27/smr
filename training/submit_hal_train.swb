#!/bin/bash

#SBATCH --job-name="run_gpux2"
#SBATCH --output="run_gpux2.%j.%N.out"

#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --export=ALL
#SBATCH -t 72:00:00
#SBATCH --exclude=hal14
#SBATCH --reservation=khan74_70

#module load powerai
#conda activate /home/khan74/.conda/envs/hvd
#mpirun -np $SLURM_NTASKS python train_cooley.py --batch_size 8


#module load powerai
module load wmlce
echo $PYTHONPATH
module list
conda activate /home/ptyagi3/.conda/envs/hvd
NODE_LIST=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:4,/g' )
NODE_LIST=${NODE_LIST%?}
echo $NODE_LIST
python --version
mpirun -np 16 -H $NODE_LIST -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^lo -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_openib_verbose 1 -mca btl_tcp_if_incle 192.168.0.0/16 -mca oob_tcp_if_include 192.168.0.0/16 python train_cooley.py --batch_size 4
