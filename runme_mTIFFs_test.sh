#!/bin/sh

SCRIPT_NAME="Test"
NUM=2
JOBLIST=""

# rm -f -r ${PALM_DATA_DIR}/${TEMP_DIR}
# mkdir ${PALM_DATA_DIR}/${TEMP_DIR}
# cd ${PALM_DATA_DIR}/${TEMP_DIR}


for i in `seq 0 $((NUM-1))`;
  do
    echo "#!/bin/sh" 1>${SCRIPT_NAME}${i}.sh
    echo "hostname">>${SCRIPT_NAME}${i}.sh
    echo 'echo $DISPLAY'>>${SCRIPT_NAME}${i}.sh
    echo 'echo “removing display variable”'>>${SCRIPT_NAME}${i}.sh
    echo 'unset $DISPLAY'>>${SCRIPT_NAME}${i}.sh
    
done

