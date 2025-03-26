#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

echo "checking required directories exist"
if [[ ! -d ${OUTPUT_DIR}/output6/ ]]; then
    log_error "Directory ${OUTPUT_DIR}/output5/ does not exist."
fi


echo "# Running the factScore.py"
if ! python3 factScore.py; then
    log_error "factScore failed to execute. Please check the script and inputs."
fi

echo "moving output to next step"


mv WikiRC_ESO.json ${INPUT_DIR}/output5/smryCmp/
mv factScore.json ${INPUT_DIR}/output5/

#moving only relevant part to next stage

find ${INPUT_DIR}/input6/output5/ -mindepth 1 -maxdepth 1 ! -name 'factScore' -exec mv {} ${OUTPUT_DIR}/output6/ \;
