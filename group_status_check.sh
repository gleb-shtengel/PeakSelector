#!/bin/sh
PROG_NAME="GroupPeaksWorker.sav"
MONITOR_NAME="read_out_files.sav"
SCRIPT_NAME="Grouping"
PALM_DATA_DIR=$2
IDL_SCR_DIR=$3
TEMP_DIR="temp_shells/"

NUM=$1
USER=$(whoami)
JOBLIST=""

for i in `seq 0 $((NUM-1))`;
  do
    JOBLIST=$JOBLIST${SCRIPT_NAME}${i},
done

cd ${IDL_SCR_DIR}
#source /usr/local/rsi/idl/bin/idl_setup.bash
source /misc/local/exelis/idl/bin/idl_setup.bash
idl -rt=${IDL_SCR_DIR}/${MONITOR_NAME} -arg ${PALM_DATA_DIR}/${TEMP_DIR}

qsub -pe batch 2 -l h_rt=3599 -N last -o /dev/null -j y -b y -cwd -V -hold_jid $JOBLIST
echo "finished cluster processing"

cd ${PALM_DATA_DIR}
rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
