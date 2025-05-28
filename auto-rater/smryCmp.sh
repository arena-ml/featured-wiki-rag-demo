#!/bin/bash
set -euo pipefail  # Exit on error, treat unset variables as errors, and fail on pipeline errors

# Logging function for error messages
log_error() {
    echo "[ERROR] $1" >&2
    exit 1
}


echo "Running the summary comparsion script"
if ! python3 smryCmp.py; then
    log_error " SmryCmp.py failed to execute. Please check the script and inputs."
fi


echo "moving output to next step"



#moving only relevant part to next stage
mv ${INPUT_DIR}/input6/output5/smryCmp/ ${OUTPUT_DIR}/output6/
