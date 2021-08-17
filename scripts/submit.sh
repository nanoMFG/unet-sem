#!/bin/bash

#SBATCH -N 1
#SBATCH -J unet-sem
#SBATCH -p GPU
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:8


# Shared project directory for gresq team
SHARED=/ocean/projects/asc190029p/shared
# Input data
DATADIR=$SHARED/GRRESQ/data
# Code repo
CODENAME='unet-sem'
#CODEDIR=${HOME}/${CODENAME}
CODEDIR=${SHARED}/${CODENAME}
SUBMITDIR=`pwd`

# Get some version information from the unet code repo
cd ${CODEDIR}
# BRANCH=`git branch | sed -e 's/^\*\ //' | sed -e 's/$//'`
RD=`git log -n 1 --pretty=format:"%cD"`
RDATE=`date -d "$RD" +%m-%d-%Y-%T`
RHASH=`git log -n 1 --pretty=format:"%h" | sed -e 's/$//'`
VERSION=${RHASH}_${RDATE}
cd ${SUBMITDIR}

# Create a unique run directory name
DIRNAME=${SLURM_JOB_NAME}-${SLURM_JOB_ID}
# Setup the run location
RUNDIR=${CODENAME}/${VERSION}/${DIRNAME}
LOCALDIR=${LOCAL}/${RUNDIR}     # RUNDIR on node-local filesystem for bridges-2
# Setup the save location
SAVEDIR=${PROJECT}/${RUNDIR} # Directory to copy results to

set -x
mkdir -p ${SAVEDIR}
mkdir -p ${LOCALDIR}
cp -r -p ${CODEDIR}/* ${LOCALDIR}

module load AI/anaconda3-tf2.2020.11
source activate $AI_ENV
cd ${LOCALDIR}
echo "SAVEDIR: ${SAVEDIR}"
#python main.py --augment --ngpu 4 --batch_size 16 --nepochs 10 2>&1 | tee out.log
#python main.py --augment --ngpu 4 --batch_size 16 --nepochs 10 --input_size 512 2>&1 | tee out.log
#python main.py --augment --ngpu 4 --batch_size 8  --nepochs 20 --input_size 512 2>&1 | tee out.log
#cp -R -p out.log output *.hdf5  ${SAVEDIR}
#python main.py --ngpu 4 --batch_size 16 --nepochs 10 --input_size 512 2>&1 | tee out.log
python main.py --lr 1e-5 --augment --ngpu 4 --batch_size 16 \
	--nepochs 50 --input_size 512 --input_dir $DATADIR 2>&1 | tee out.log

# Copy data back to save location
cp -R -p * ${SAVEDIR}

set +x
#echo $RUNDIR
#echo $SAVEDIR
