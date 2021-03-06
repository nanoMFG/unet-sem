#!/bin/bash

#SBATCH -N 1
#SBATCH -J unet-sem
#SBATCH -p GPU
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:8

CODENAME='unet-sem'
DIRNAME=${SLURM_JOB_NAME}-${SLURM_JOB_ID}
CODEDIR=${HOME}/${CODENAME}
SUBMITDIR=`pwd`

cd ${CODEDIR}
# BRANCH=`git branch | sed -e 's/^\*\ //' | sed -e 's/$//'`
RD=`git log -n 1 --pretty=format:"%cD"`
RDATE=`date -d "$RD" +%m-%d-%Y-%T`
RHASH=`git log -n 1 --pretty=format:"%h" | sed -e 's/$//'`
VERSION=${RHASH}_${RDATE}
cd ${SUBMITDIR}

RUNDIR=${CODENAME}/${VERSION}/${DIRNAME}
SCRATCHDIR=${PROJECT}/${RUNDIR}
LOCALDIR=${LOCAL}/${RUNDIR}

set -x
mkdir -p ${SCRATCHDIR}
mkdir -p ${LOCALDIR}
cp -r -p ${CODEDIR}/* ${LOCALDIR}

module load AI/anaconda3-tf2.2020.11
source activate $AI_ENV
cd ${LOCALDIR}
echo "SCRATCHDIR: ${SCRATCHDIR}"
#python main.py --augment --ngpu 4 --batch_size 16 --nepochs 10 2>&1 | tee out.log
#python main.py --augment --ngpu 4 --batch_size 16 --nepochs 10 --input_size 512 2>&1 | tee out.log
#python main.py --augment --ngpu 4 --batch_size 8  --nepochs 20 --input_size 512 2>&1 | tee out.log
#cp -R -p out.log output *.hdf5  ${SCRATCHDIR}
#python main.py --ngpu 4 --batch_size 16 --nepochs 10 --input_size 512 2>&1 | tee out.log
python main.py --lr 1e-5 --augment --ngpu 8 --batch_size 16 --nepochs 50 --input_size 512 2>&1 | tee out.log
cp -R -p * ${SCRATCHDIR}

set +x
#echo $RUNDIR
#echo $SAVEDIR
