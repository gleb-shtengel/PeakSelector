#!/bin/sh
PROG_NAME="TransformRaw_Save_SaveSum_Worker.sav"
MONITOR_NAME="read_out_files.sav"
SCRIPT_NAME="Transform"
PALM_DATA_DIR=$2
IDL_SCR_DIR=$3
TEMP_DIR="temp_shells/"

NUM=$1
USER=$(whoami)
JOBLIST=""

rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
mkdir ${PALM_DATA_DIR}/${TEMP_DIR}
cd ${PALM_DATA_DIR}/${TEMP_DIR}

echo "number of loops-1="$((NUM-1))

for i in `seq 0 $((NUM-1))`;
  do
    echo "#!/bin/sh" 1>${SCRIPT_NAME}${i}.sh
    echo "hostname">>${SCRIPT_NAME}${i}.sh
    echo "cd ${PALM_DATA_DIR}" 1>>${SCRIPT_NAME}${i}.sh
    # echo "source /usr/local/rsi/idl/bin/idl_setup.bash" 1>>${SCRIPT_NAME}${i}.sh old version of IDL
    echo "source /misc/local/exelis/idl/bin/idl_setup.bash" 1>>${SCRIPT_NAME}${i}.sh
    echo "idl -rt=${IDL_SCR_DIR}/${PROG_NAME} -args "${i}" "${PALM_DATA_DIR} 1>>${SCRIPT_NAME}${i}.sh
     chmod +x ${SCRIPT_NAME}${i}.sh
    qsub -l sl7=true -cwd -pe batch 2 -l h_rt=3599 -V -N ${SCRIPT_NAME}${i} -j y -o ${SCRIPT_NAME}${i}.out -b y -l idl_rt=6 ${PALM_DATA_DIR}/${TEMP_DIR}${SCRIPT_NAME}${i}.sh
    JOBLIST=$JOBLIST${SCRIPT_NAME}${i},
done

cd ${IDL_SCR_DIR}
#source /usr/local/rsi/idl/bin/idl_setup.bash
source /misc/local/exelis/idl/bin/idl_setup.bash
idl -rt=${IDL_SCR_DIR}/${MONITOR_NAME} -arg ${PALM_DATA_DIR}/${TEMP_DIR}

qsub -l sl7=true -pe batch 2 -l h_rt=3599 -N last -o /dev/null -j y -b y -cwd -V -hold_jid $JOBLIST
echo "finished cluster processing (data transformation)"

cd ${PALM_DATA_DIR}
rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
