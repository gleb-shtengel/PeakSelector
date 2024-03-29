#!/bin/sh
export DISPLAY=hostname:0.0
PROG_NAME="GroupPeaksWorker.sav"
MONITOR_NAME="read_out_files.sav"
SCRIPT_NAME="Grouping"
PALM_DATA_DIR=$2
IDL_SCR_DIR=$3
#TEMP_DIR="temp_shells/"
TEMP_FOLDER=$4
TEMP_DIR=$TEMP_FOLDER
TEMP_DIR+="/temp_shells/"

NUM=$1
USER=$(whoami)

#cd ${PALM_DATA_DIR}/${TEMP_DIR}
cd ${TEMP_DIR}

echo "#!/bin/sh" 1>${SCRIPT_NAME}${NUM}.sh
echo "hostname">>${SCRIPT_NAME}${NUM}.sh
echo "cd ${PALM_DATA_DIR}" 1>>${SCRIPT_NAME}${NUM}.sh
#echo "source /usr/local/rsi/idl/bin/idl_setup.bash" 1>>${SCRIPT_NAME}${NUM}.sh  old version of IDL
echo "source /misc/local/exelis/idl/bin/idl_setup.bash" 1>>${SCRIPT_NAME}${NUM}.sh
echo "idl -rt=${IDL_SCR_DIR}/${PROG_NAME} -args "${NUM}" "${PALM_DATA_DIR}" "${TEMP_FOLDER} 1>>${SCRIPT_NAME}${NUM}.sh
 chmod +x ${SCRIPT_NAME}${NUM}.sh
#qsub -cwd -pe batch 4 -l d_rt=3599 -V -N ${SCRIPT_NAME}${NUM} -j y -o ${SCRIPT_NAME}${NUM}.out -b y -l idl_rt=6 ${PALM_DATA_DIR}/${TEMP_DIR}${SCRIPT_NAME}${NUM}.sh
bsub -n 4 -R"rusage[idl_rt=6]" -J ${SCRIPT_NAME}${NUM} -o ${SCRIPT_NAME}${NUM}.out idl_rt=6 ${TEMP_DIR}${SCRIPT_NAME}${NUM}.sh