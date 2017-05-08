#!/bin/sh
PROG_NAME="GroupPeaksWorker.sav"
MONITOR_NAME="read_out_files.sav"
SCRIPT_NAME="Grouping"
PALM_DATA_DIR=$2
IDL_SCR_DIR=$3
TEMP_DIR="temp_shells/"

NUM=$1
USER=$(whoami)

rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
mkdir ${PALM_DATA_DIR}/${TEMP_DIR}
cd ${PALM_DATA_DIR}/${TEMP_DIR}
