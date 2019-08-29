#!/bin/bash
jobid=$1
echo $jobid
outfile=`ls -1 | grep $jobid`
SCRATCHDIR=`cat $outfile | grep SCRATCHDIR | grep -v echo | sed -e 's/SCRATCHDIR:\ //' | sed -e 's/$//'`
WD=`pwd`

echo $SCRATCHDIR
#cd $SCRATCHDIR
#rm -rf __pycache__/ README.md data/ *.py
#sacct -j $jobid --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
#cd $WD > stats.log

#module load uberftp
#uberftp -u dadams@illinois.edu ftp.box.com -p 1hatFdc9927 "put -r $SCRATCHDIR"

