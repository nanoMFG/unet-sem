#!/bin/bash
jobid=$1
echo $jobid
outfile=`ls -1 | grep $jobid`
SCRATCHDIR=`cat slurm-${jobid}.out | grep SCRATCHDIR | grep -v echo | sed -e 's/SCRATCHDIR:\ //' | sed -e 's/$//'`
WD=`pwd`

echo $SCRATCHDIR

echo "'$2'"
if [ "$2" == '1' ] ; then
echo "Performing post operations"
cp slurm-${jobid}.out $SCRATCHDIR
cd $SCRATCHDIR
rm -rf __pycache__/ README.md data/
sacct -j $jobid --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist > stats.log
cd $WD
fi

#module load uberftp
#uberftp -u dadams@illinois.edu ftp.box.com -p 1hatFdc9927 "put -r $SCRATCHDIR"

