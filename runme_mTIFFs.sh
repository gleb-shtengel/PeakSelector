#!/bin/sh
PROG_NAME="ReadRawWorker_mTIFFs.sav"
MONITOR_NAME="read_out_files.sav"
SCRIPT_NAME="Palm"
PALM_DATA_DIR=$2
IDL_SCR_DIR=$3
#TEMP_DIR="temp_shells/"
TEMP_FOLDER=$4
TEMP_DIR=$TEMP_FOLDER
TEMP_DIR+="/temp_shells/"

NUM=$1
USER=$(whoami)
JOBLIST=""

# rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
# mkdir ${PALM_DATA_DIR}/${TEMP_DIR}
# cd ${PALM_DATA_DIR}/${TEMP_DIR}
rm -f -r ${TEMP_DIR}
mkdir ${TEMP_DIR}
cd ${TEMP_DIR}


for i in `seq 0 $((NUM-1))`;
  do
    echo "#!/bin/sh" 1>${SCRIPT_NAME}${i}.sh
    echo "hostname">>${SCRIPT_NAME}${i}.sh
    echo 'echo $DISPLAY'>>${SCRIPT_NAME}${i}.sh
    echo 'echo “removing display variable”'>>${SCRIPT_NAME}${i}.sh
    echo 'unset DISPLAY'>>${SCRIPT_NAME}${i}.sh
    echo "cd ${PALM_DATA_DIR}" 1>>${SCRIPT_NAME}${i}.sh
    # echo "source /usr/local/rsi/idl/bin/idl_setup.bash" 1>>${SCRIPT_NAME}${i}.sh old version of IDL
    echo "source /misc/local/exelis/idl/bin/idl_setup.bash" 1>>${SCRIPT_NAME}${i}.sh
    echo "idl -rt=${IDL_SCR_DIR}/${PROG_NAME} -args "${i}" "${PALM_DATA_DIR}" "${TEMP_FOLDER} 1>>${SCRIPT_NAME}${i}.sh
     chmod +x ${SCRIPT_NAME}${i}.sh
    #qsub -cwd -pe batch 2 -l d_rt=9999 -V -N ${SCRIPT_NAME}${i} -j y -o ${SCRIPT_NAME}${i}.out -b y -l idl_rt=6 ${PALM_DATA_DIR}/${TEMP_DIR}${SCRIPT_NAME}${i}.sh
    bsub -n 2 -R"rusage[idl_rt=6]" -J ${SCRIPT_NAME}${i} -o ${SCRIPT_NAME}${i}.out idl_rt=6 ${TEMP_DIR}${SCRIPT_NAME}${i}.sh
    JOBLIST=$JOBLIST${SCRIPT_NAME}${i},
done

cd ${IDL_SCR_DIR}
#source /usr/local/rsi/idl/bin/idl_setup.bash
source /misc/local/exelis/idl/bin/idl_setup.bash
#idl -rt=${IDL_SCR_DIR}/${MONITOR_NAME} -arg ${PALM_DATA_DIR}/${TEMP_DIR}
idl -rt=${IDL_SCR_DIR}/${MONITOR_NAME} -arg ${TEMP_DIR}

#qsub -pe batch 1 -l d_rt=9999 -N last -o /dev/null -j y -b y -cwd -V -hold_jid $JOBLIST
#bsub -n 1 -J last -o /dev/null -w $JOBLIST

echo "finished cluster processing"

# cd ${PALM_DATA_DIR}/${TEMP_DIR}
# rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
cd ${TEMP_DIR}
rm -f -r ${TEMP_DIR}
